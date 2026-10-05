"""Sonata-style self-distillation for LitePT on ALS cookies ("SonataLite").

Reference: Wu et al. 2025, "Sonata: Self-Supervised Learning of Reliable Point Representations"
(Pointcept). The recipe is simplified for single-cookie aerial lidar:

* student and EMA teacher share the LitePT encoder architecture (enc_mode=True, no decoder; the
  decoder-free design avoids Sonata's "geometric shortcut").
* teacher encodes the 2 unmasked global views; student encodes the 2 global views with grid masking
  plus the local views.
* point-level loss: encoder features are "up-cast" (concatenated back through `upcast_levels` pooling
  stages); student points are matched to teacher points of the same sample via their pre-augmentation
  coordinates (`origin_coord`, hashed at `match_grid` meters); prototype-head distributions are
  distilled with Sinkhorn-Knopp-normalized teacher targets.
* scene-level loss (DINO): mean-pooled deepest-stage features of every student view predict the
  teacher's pooled global-view distribution of the other global views. This gives the per-cookie
  embedding that the downstream classification task needs.
"""

import copy
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from litept.model import LitePT, Point, offset2batch


class DINOHead(nn.Module):
    """MLP -> L2-normalized bottleneck -> weight-normalized prototypes (as in DINO / Sonata)."""

    def __init__(self, in_dim, num_prototypes=4096, hidden_dim=2048, bottleneck_dim=256, nlayers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(nlayers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers += [nn.Linear(hidden_dim, bottleneck_dim)]
        self.mlp = nn.Sequential(*layers)
        self.prototypes = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, num_prototypes, bias=False))
        self.prototypes.parametrizations.weight.original0.data.fill_(1)
        self.prototypes.parametrizations.weight.original0.requires_grad = False

    def forward(self, x):
        x = F.normalize(self.mlp(x), dim=-1, p=2)
        return self.prototypes(x)


@torch.no_grad()
def sinkhorn_knopp(logits, temp, n_iters=3):
    """Sinkhorn-Knopp centering of teacher logits [N, K] -> soft targets [N, K] (DDP-aware)."""
    q = torch.exp((logits.float() - logits.float().max()) / temp).t()  # [K, N]
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    n = torch.tensor([q.shape[1]], device=q.device, dtype=torch.float)
    if world > 1:
        dist.all_reduce(n)
    k = q.shape[0]
    s = q.sum()
    if world > 1:
        dist.all_reduce(s)
    q /= s
    for _ in range(n_iters):
        r = q.sum(dim=1, keepdim=True)
        if world > 1:
            dist.all_reduce(r)
        q /= r.clamp_min(1e-12)
        q /= k
        q /= q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        q /= n
    q *= n
    return q.t()


def upcast(point, levels):
    """Concatenate deeper features back onto `levels` finer pooling stages (Sonata feature up-casting)."""
    for _ in range(levels):
        if "pooling_parent" not in point.keys():
            break
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    return point


def segment_mean(feat, offset):
    """Mean of features per batch element given LitePT offsets."""
    batch = offset2batch(offset)
    out = torch.zeros(len(offset), feat.shape[1], device=feat.device, dtype=feat.dtype)
    out.index_add_(0, batch, feat)
    counts = torch.diff(offset, prepend=offset.new_zeros(1)).clamp_min(1).to(feat.dtype)
    return out / counts[:, None]


class ALSEncoder(nn.Module):
    """LitePT encoder with an optional learnable mask token (used by the SSL student)."""

    def __init__(self, backbone_cfg, mask_token=True):
        super().__init__()
        cfg = dict(backbone_cfg)
        cfg.pop("type", None)
        cfg["enc_mode"] = True
        self.backbone = LitePT(**cfg)
        self.out_channels = cfg["enc_channels"]
        self.mask_token = nn.Parameter(torch.zeros(1, cfg["enc_channels"][0])) if mask_token else None
        if self.mask_token is not None:
            nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, data, mask=None):
        """data: dict with coord, grid_coord, feat, offset (+ origin_coord). Returns deepest-stage Point."""
        bb = self.backbone
        if mask is not None:
            data = dict(data)
            data["feat"] = data["feat"] * (~mask).unsqueeze(-1).to(data["feat"].dtype)
        point = Point(data)
        if bb.enc_attn[0]:
            point.serialization(order=bb.order, shuffle_orders=bb.shuffle_orders)
        point.sparsify()
        point = bb.embedding(point)
        if mask is not None and self.mask_token is not None:
            point.feat = torch.where(mask.unsqueeze(-1), self.mask_token.to(point.feat.dtype), point.feat)
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        point = bb.enc(point)
        return point


class SonataLite(nn.Module):
    def __init__(
        self,
        backbone,
        upcast_levels=2,
        num_prototypes=4096,
        scene_prototypes=4096,
        head_hidden=2048,
        head_bottleneck=256,
        mask_grid=6.0,
        mask_ratio=(0.4, 0.7),
        match_grid=2.0,
        teacher_temp=(0.04, 0.07),
        teacher_temp_warmup=0.1,
        student_temp=0.1,
        momentum=(0.994, 1.0),
        scene_weight=0.5,
        point_weight=1.0,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student = ALSEncoder(backbone, mask_token=True)
        ch = self.student.out_channels
        point_dim = sum(ch[len(ch) - 1 - upcast_levels:])
        self.student_point_head = DINOHead(point_dim, num_prototypes, head_hidden, head_bottleneck)
        self.student_scene_head = DINOHead(ch[-1], scene_prototypes, head_hidden, head_bottleneck)
        self.teacher = copy.deepcopy(self.student)
        self.teacher_point_head = copy.deepcopy(self.student_point_head)
        self.teacher_scene_head = copy.deepcopy(self.student_scene_head)
        for m in (self.teacher, self.teacher_point_head, self.teacher_scene_head):
            for p in m.parameters():
                p.requires_grad = False
        self.upcast_levels = upcast_levels
        self.mask_grid, self.mask_ratio, self.match_grid = mask_grid, mask_ratio, match_grid
        self.teacher_temp, self.teacher_temp_warmup = teacher_temp, teacher_temp_warmup
        self.student_temp = student_temp
        self.momentum = momentum
        self.scene_weight, self.point_weight = scene_weight, point_weight
        # scene-level targets use DINO centering (few views per batch -> Sinkhorn would be degenerate)
        self.center_momentum = center_momentum
        self.register_buffer("scene_center", torch.zeros(1, scene_prototypes))

    # ---------- helpers ----------
    def backbone_state_dict(self, which="teacher"):
        """LitePT weights for downstream use (teacher = EMA weights, usually the better encoder)."""
        enc = self.teacher if which == "teacher" else self.student
        return enc.backbone.state_dict()

    def student_modules(self):
        return [self.student, self.student_point_head, self.student_scene_head]

    @torch.no_grad()
    def update_teacher(self, progress):
        m = self.momentum[1] - (self.momentum[1] - self.momentum[0]) * (math.cos(math.pi * progress) + 1) / 2
        for s, t in zip(self.student_modules(), (self.teacher, self.teacher_point_head, self.teacher_scene_head)):
            for ps, pt in zip(s.parameters(), t.parameters()):
                pt.data.mul_(m).add_(ps.detach().data, alpha=1 - m)
        return m

    @torch.no_grad()
    def _update_center(self, logits):
        mean = logits.mean(0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(mean)
            mean /= dist.get_world_size()
        self.scene_center.mul_(self.center_momentum).add_(mean, alpha=1 - self.center_momentum)

    def _teacher_temp(self, progress):
        t0, t1 = self.teacher_temp
        if progress >= self.teacher_temp_warmup:
            return t1
        return t0 + (t1 - t0) * progress / self.teacher_temp_warmup

    @staticmethod
    def _view(batch, prefix):
        return dict(
            coord=batch[f"{prefix}_coord"],
            grid_coord=batch[f"{prefix}_grid_coord"],
            feat=batch[f"{prefix}_feat"],
            offset=batch[f"{prefix}_offset"],
            origin_coord=batch[f"{prefix}_origin_coord"],
        )

    def _grid_mask(self, view, ratio):
        """Mask whole mask_grid x mask_grid (xy) columns of each view with probability `ratio`."""
        oc = view["origin_coord"]
        batch = offset2batch(view["offset"])
        cell = torch.floor(oc[:, :2] / self.mask_grid).long()
        key = (batch * 1_000_003 + cell[:, 0]) * 1_000_003 + cell[:, 1]
        _, inv = torch.unique(key, return_inverse=True)
        masked_cells = torch.rand(int(inv.max()) + 1, device=oc.device) < ratio
        return masked_cells[inv]

    def _match_keys(self, origin_coord, sample_id):
        cell = torch.floor(origin_coord / self.match_grid).long() + 1_000
        return ((sample_id * 4096 + cell[:, 0]) * 4096 + cell[:, 1]) * 4096 + cell[:, 2]

    def _encode(self, encoder, view, mask=None):
        point = encoder(view, mask=mask)
        scene_feat = segment_mean(point.feat, point.offset)  # per view
        point_up = upcast(point, self.upcast_levels)
        return point_up, scene_feat

    # ---------- forward ----------
    def forward(self, batch, progress=0.0):
        n_global = batch["global_offset"].shape[0]
        n_local = batch["local_offset"].shape[0] if "local_offset" in batch else 0
        bsz = batch["num_samples"]
        g_per, l_per = n_global // bsz, n_local // bsz
        gview, lview = self._view(batch, "global"), (self._view(batch, "local") if n_local else None)
        t_temp = self._teacher_temp(progress)

        # teacher: unmasked global views
        with torch.no_grad():
            t_point, t_scene = self._encode(self.teacher, gview)
            t_logits = self.teacher_point_head(t_point.feat)
            t_probs = sinkhorn_knopp(t_logits, t_temp)
            t_scene_logits = self.teacher_scene_head(t_scene).float()
            t_scene_probs = F.softmax((t_scene_logits - self.scene_center) / t_temp, dim=-1)
            self._update_center(t_scene_logits)
            # aggregate teacher point targets per (sample, match cell)
            t_sample = offset2batch(t_point.offset) // g_per
            t_keys = self._match_keys(t_point.origin_coord, t_sample)
            uniq, inv = torch.unique(t_keys, return_inverse=True)
            targets = torch.zeros(len(uniq), t_probs.shape[1], device=t_probs.device)
            targets.index_add_(0, inv, t_probs.float())
            targets /= torch.bincount(inv, minlength=len(uniq)).clamp_min(1)[:, None]

        ratio = self.mask_ratio[0] + (self.mask_ratio[1] - self.mask_ratio[0]) * min(progress / 0.5, 1.0)
        gmask = self._grid_mask(gview, ratio)
        s_views = [("global", gview, gmask, g_per)]
        if lview is not None:
            s_views.append(("local", lview, None, l_per))

        losses, point_losses, scene_losses = {}, [], []
        for name, view, mask, per in s_views:
            s_point, s_scene = self._encode(self.student, view, mask=mask)
            # point-level distillation
            s_sample = offset2batch(s_point.offset) // per
            s_keys = self._match_keys(s_point.origin_coord, s_sample)
            pos = torch.searchsorted(uniq, s_keys).clamp_max(len(uniq) - 1)
            hit = uniq[pos] == s_keys
            if hit.any():
                s_logits = self.student_point_head(s_point.feat[hit])
                loss = -(targets[pos[hit]] * F.log_softmax(s_logits.float() / self.student_temp, -1)).sum(-1).mean()
                point_losses.append(loss)
                losses[f"point_{name}"] = loss.detach()
                losses[f"match_{name}"] = hit.float().mean().detach()
            # scene-level distillation: every student view vs teacher global views of the same sample
            s_scene_logp = F.log_softmax(self.student_scene_head(s_scene).float() / self.student_temp, -1)
            s_scene_logp = s_scene_logp.view(bsz, per, -1)
            t_sp = t_scene_probs.view(bsz, g_per, -1)
            terms = []
            for i in range(per):
                for j in range(g_per):
                    if name == "global" and i == j:
                        continue  # same view
                    terms.append(-(t_sp[:, j] * s_scene_logp[:, i]).sum(-1).mean())
            if terms:
                scene_losses.append(torch.stack(terms).mean())
                losses[f"scene_{name}"] = scene_losses[-1].detach()

        loss = 0.0
        if point_losses:
            loss = loss + self.point_weight * torch.stack(point_losses).mean()
        if scene_losses:
            loss = loss + self.scene_weight * torch.stack(scene_losses).mean()
        losses["loss"] = loss
        losses["mask_ratio"] = torch.tensor(ratio)
        losses["teacher_temp"] = torch.tensor(t_temp)
        return losses
