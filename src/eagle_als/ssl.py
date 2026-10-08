"""Self-supervised pre-training of LitePT on ALS cookies, mirroring ForPT / Sonata.

ForPT (Yue et al. 2026, "Toward a foundation model for forest point clouds", arXiv:2609.24787)
pre-trains LitePT-S with the Sonata self-distillation recipe (Wu et al. 2025). Their code is not
public, so this is a close port of Sonata's reference implementation
(Pointcept `pointcept/models/sonata/sonata_v1m1_base.py`, commit 1342eda), with the settings
ForPT states explicitly. Both losses are point-level:

* masked-to-global (ForPT L_m2g = Sonata `mask_loss` + `roll_mask_loss`): the student encodes the
  two masked global views; each student point is matched (1-NN on pre-augmentation coordinates,
  within `match_max_r`) to the teacher's unmasked encoding of the same view and of the other view.
* local-to-global (ForPT L_l2g = Sonata `unmask_loss`): student local-view points are matched to
  the teacher's principal global view.
* teacher targets are Sinkhorn-Knopp normalized prototype scores; features are up-cast through
  `up_cast_level` pooling stages before the heads (OnlineCluster).
* Sinkhorn memory bank (Vernata / SwAV queue): each normalization spans `sinkhorn_cookies` cookies
  over all GPUs, whatever the GPU count and accumulation. Each GPU queues the teacher head embeddings
  of recent cookies (`sinkhorn_points_per_cookie` sampled target points each, weighted so a queued
  cookie carries the same Sinkhorn mass as a current one) and re-scores them with the current
  prototypes. With gradient accumulation the queue is mostly the rest of the same optimizer step.

ALS-specific choices kept on purpose (see scripts/litePT/README.md): 0.4 m grid, larger masks
(xy columns of `mask_size` m instead of 3D cubes when `mask_dims=2`), extra lidar input channels,
and density thinning (data pipeline).
"""

from itertools import chain

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from timm.layers import trunc_normal_

from litept.model import LitePT, Point, offset2batch


def batch2offset(batch):
    return torch.cumsum(torch.bincount(batch), dim=0).long()


def offset2bincount(offset):
    return torch.diff(offset, prepend=offset.new_zeros(1))


def world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def cosine_schedule(step, total, base, final, start=None, warmup=0):
    """Value of Pointcept's CosineScheduler at `step` (linear warmup start->base, cosine base->final)."""
    if step >= total:
        return final
    if step < warmup:
        return start + (base - start) * step / max(warmup - 1, 1)
    n = total - warmup
    return final + 0.5 * (base - final) * (1 + np.cos(np.pi * (step - warmup) / n))


class OnlineCluster(nn.Module):
    """Sonata projection head: MLP -> l2 normalize -> weight-normalized prototypes."""

    def __init__(self, in_channels, hidden_channels=4096, embed_channels=256, num_prototypes=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, embed_channels),
        )
        self.apply(self._init_weights)
        self.prototype = torch.nn.utils.parametrizations.weight_norm(
            nn.Linear(embed_channels, num_prototypes, bias=False)
        )
        self.prototype.parametrizations.weight.original0.data.fill_(1)
        self.prototype.parametrizations.weight.original0.requires_grad = False

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def embed(self, feat):
        feat = self.mlp(feat)
        eps = 1e-6 if feat.dtype == torch.float16 else 1e-12
        return F.normalize(feat, dim=-1, p=2, eps=eps)

    def forward(self, feat):
        return self.prototype(self.embed(feat))


def build_backbone(cfg, **overrides):
    cfg = dict(cfg)
    cfg.pop("type", None)
    cfg.update(overrides)
    return LitePT(**cfg)


class SonataLitePT(nn.Module):
    def __init__(
        self,
        backbone,
        head_in_channels,
        head_hidden_channels=4096,
        head_embed_channels=256,
        head_num_prototypes=4096,
        teacher_custom=None,
        num_global_view=2,
        num_local_view=4,
        mask_dims=2,
        mask_size_start=2.0,
        mask_size_base=6.0,
        mask_size_warmup_ratio=0.05,
        mask_ratio_start=0.3,
        mask_ratio_base=0.7,
        mask_ratio_warmup_ratio=0.05,
        mask_jitter=None,
        teacher_temp_start=0.04,
        teacher_temp_base=0.07,
        teacher_temp_warmup_ratio=0.05,
        student_temp=0.1,
        mask_loss_weight=2 / 8,
        roll_mask_loss_weight=2 / 8,
        unmask_loss_weight=4 / 8,
        momentum_base=0.994,
        momentum_final=0.994,
        match_max_r=6.4,
        up_cast_level=2,
        grid_size=0.4,
        sinkhorn_cookies=0,
        sinkhorn_points_per_cookie=2048,
    ):
        super().__init__()
        assert num_global_view == 2, "roll mask loss (ForPT cross-view pairs) needs exactly two global views"
        assert mask_dims in (2, 3)
        self.num_global_view, self.num_local_view = num_global_view, num_local_view
        self.mask_dims = mask_dims
        self.mask_size_cfg = (mask_size_start, mask_size_base, mask_size_warmup_ratio)
        self.mask_ratio_cfg = (mask_ratio_start, mask_ratio_base, mask_ratio_warmup_ratio)
        self.teacher_temp_cfg = (teacher_temp_start, teacher_temp_base, teacher_temp_warmup_ratio)
        self.momentum_cfg = (momentum_base, momentum_final)
        self.mask_jitter = mask_jitter
        self.student_temp = student_temp
        self.mask_loss_weight = mask_loss_weight
        self.roll_mask_loss_weight = roll_mask_loss_weight
        self.unmask_loss_weight = unmask_loss_weight
        self.match_max_r = match_max_r
        self.up_cast_level = up_cast_level
        self.grid_size = grid_size
        self.sk_cookies, self.sk_points = sinkhorn_cookies, sinkhorn_points_per_cookie
        self.queues = {}  # head -> dict(emb (cookies, points, C), weight (cookies, points), pos, filled); not saved
        self.set_step(0, 1)

        head = lambda: OnlineCluster(head_in_channels, head_hidden_channels, head_embed_channels, head_num_prototypes)
        self.student = nn.ModuleDict(dict(backbone=build_backbone(backbone), mask_head=head(), unmask_head=head()))
        # teacher: same architecture, drop path etc. turned off
        self.teacher = nn.ModuleDict(
            dict(backbone=build_backbone(backbone, **(teacher_custom or {})), mask_head=head(), unmask_head=head())
        )
        for k in self.student:
            self.teacher[k].load_state_dict(self.student[k].state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

    # ---------------- schedules (Sonata before_step) ----------------
    def set_step(self, step, total):
        s0, s1, sw = self.mask_size_cfg
        r0, r1, rw = self.mask_ratio_cfg
        t0, t1, tw = self.teacher_temp_cfg
        self.mask_size = cosine_schedule(step, total, s1, s1, s0, int(total * sw))
        self.mask_ratio = cosine_schedule(step, total, r1, r1, r0, int(total * rw))
        self.teacher_temp = cosine_schedule(step, total, t1, t1, t0, int(total * tw))
        self.momentum = cosine_schedule(step, total, *self.momentum_cfg)

    @torch.no_grad()
    def update_teacher(self):
        """EMA update (Sonata after_step)."""
        m = self.momentum
        s = list(self.student.parameters())
        t = list(self.teacher.parameters())
        torch._foreach_mul_(t, m)
        torch._foreach_add_(t, s, alpha=1 - m)
        return m

    def backbone_state_dict(self, which="teacher"):
        return (self.teacher if which == "teacher" else self.student)["backbone"].state_dict()

    # ---------------- Sonata components ----------------
    @staticmethod
    @torch.no_grad()
    def sinkhorn_knopp(feat, temp, num_iter=3, extra=None):
        """Sinkhorn targets for `feat` (points x prototypes). `extra` = (scores, weights) of queued points
        that join the normalization with column mass `weights` (current points: 1); only the targets
        of `feat` are returned. Without `extra` this is Sonata's Sinkhorn-Knopp."""
        m = len(feat)
        if extra is None:
            q, w = feat.to(torch.float32, copy=True), None
        else:
            # one float32 buffer and in-place ops: with a queue this matrix is several GB
            q = torch.empty(m + len(extra[0]), feat.shape[1], device=feat.device, dtype=torch.float32)
            q[:m], q[m:] = feat, extra[0]
            w = torch.cat([torch.ones(m, device=feat.device), extra[1].float()])
        q = q.div_(temp).exp_().t()
        if w is not None:
            q.mul_(w)
        n = w.sum().reshape(1) if w is not None else torch.tensor([q.shape[1]], device=q.device, dtype=torch.float)
        if world_size() > 1:
            dist.all_reduce(n)
        k = q.shape[0]
        sum_q = q.sum()
        if world_size() > 1:
            dist.all_reduce(sum_q)
        q.div_(sum_q)
        for _ in range(num_iter):
            q_row_sum = q.sum(dim=1, keepdim=True)
            if world_size() > 1:
                dist.all_reduce(q_row_sum)
            q.div_(q_row_sum).div_(k)
            q.div_(q.sum(dim=0, keepdim=True))
            q.div_(n) if w is None else q.mul_(w / n)
        q.mul_(n)
        return q.t() if extra is None else q[:, :m].t().clone()  # clone frees the queue columns

    # ---------------- Sinkhorn memory bank ----------------
    @torch.no_grad()
    def queue_scores(self, name, head):
        """(scores, weights) of head `name`'s queued embeddings under the current prototypes, or None."""
        qu = self.queues.get(name)
        if qu is None or qu["filled"] == 0:
            return None
        w = qu["weight"][: qu["filled"]].flatten()
        keep = w > 0
        emb = qu["emb"][: qu["filled"]].flatten(0, 1)[keep]
        return head.prototype(emb.to(head.prototype.weight.dtype)), w[keep]

    @torch.no_grad()
    def push_queue(self, name, emb, sample):
        """Queue up to sk_points random target rows per cookie (`emb`: teacher head embeddings of the
        rows a loss used, `sample`: their cookie index in this micro-batch), each weighted by
        rows / sampled so a queued cookie keeps the Sinkhorn mass it had as a current one."""
        if not (self.training and self.sk_cookies > 0) or len(emb) == 0:
            return
        n_cookies, p = int(sample.max()) + 1, self.sk_points
        if name not in self.queues:  # sized once: per-GPU share of sk_cookies minus the micro-batch
            size = max(0, -(-self.sk_cookies // world_size()) - n_cookies)
            self.queues[name] = dict(emb=emb.new_zeros(size, p, emb.shape[1]), weight=emb.new_zeros(size, p, dtype=torch.float32),
                                     pos=0, filled=0, size=size, batch=n_cookies)
        qu = self.queues[name]
        if qu["size"] == 0:
            return
        order = torch.randperm(len(emb), device=emb.device)
        s_sorted, perm = torch.sort(sample[order], stable=True)
        order = order[perm]
        counts = torch.bincount(sample, minlength=n_cookies)
        rank = torch.arange(len(order), device=emb.device) - (torch.cumsum(counts, 0) - counts)[s_sorted]
        keep = rank < p
        rows, cookie, slot = order[keep], s_sorted[keep], rank[keep]
        new_emb = emb.new_zeros(n_cookies, p, emb.shape[1])
        new_w = torch.zeros(n_cookies, p, device=emb.device)
        new_emb[cookie, slot] = emb[rows]
        new_w[cookie, slot] = (counts.float() / counts.clamp(max=p).clamp(min=1).float())[cookie]
        new_emb, new_w = new_emb[-qu["size"]:], new_w[-qu["size"]:]
        idx = (qu["pos"] + torch.arange(len(new_emb), device=emb.device)) % qu["size"]
        qu["emb"][idx], qu["weight"][idx] = new_emb, new_w
        qu["pos"] = (qu["pos"] + len(new_emb)) % qu["size"]
        qu["filled"] = min(qu["filled"] + len(new_emb), qu["size"])

    def sinkhorn_cookies_now(self, name="mask"):
        """Cookies over all GPUs in the latest normalization of head `name` (current + queued)."""
        qu = self.queues.get(name)
        return None if qu is None else (qu["filled"] + qu["batch"]) * world_size()

    @torch.no_grad()
    def generate_mask(self, coord, offset):
        """Mask a `mask_ratio` fraction of grid patches (xy columns if mask_dims=2, else 3D cubes)."""
        batch = offset2batch(offset)
        coord = coord[:, : self.mask_dims]
        min_coord = torch_scatter.segment_coo(coord, batch, reduce="min")
        grid_coord = ((coord - min_coord[batch]) // self.mask_size).int()
        grid_coord = torch.cat([batch.unsqueeze(-1).int(), grid_coord], dim=-1)
        unique, point_cluster = torch.unique(grid_coord, dim=0, sorted=True, return_inverse=True)
        patch_num = unique.shape[0]
        mask_patch_index = torch.randperm(patch_num, device=coord.device)[: int(patch_num * self.mask_ratio)]
        return torch.isin(point_cluster, mask_patch_index)

    @torch.no_grad()
    def match_neighbour(self, view1_coord, view1_offset, view2_coord, view2_offset):
        """For each view1 point, its nearest view2 point of the same batch element within match_max_r.

        Equivalent of Sonata's pointops.knn_query(1, ...) matching (KD-tree on CPU; pointops is a
        Pointcept CUDA extension we do not build). Returns [M, 2] (view1 index, view2 index).
        """
        from scipy.spatial import cKDTree

        c1 = view1_coord.float().cpu().numpy()
        c2 = view2_coord.float().cpu().numpy()
        o1 = [0] + view1_offset.tolist()
        o2 = [0] + view2_offset.tolist()
        pairs = []
        for b in range(len(o1) - 1):
            a1, e1, a2, e2 = o1[b], o1[b + 1], o2[b], o2[b + 1]
            if e1 == a1 or e2 == a2:
                continue
            dist_, idx = cKDTree(c2[a2:e2]).query(c1[a1:e1], k=1, distance_upper_bound=self.match_max_r, workers=-1)
            keep = np.isfinite(dist_)
            pairs.append(np.stack([np.arange(a1, e1)[keep], idx[keep] + a2], axis=1))
        index = np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=np.int64)
        return torch.from_numpy(index).long().to(view1_coord.device)

    @torch.no_grad()
    def roll_point(self, point):
        """[pc1, pc2] -> [pc2, pc1] within every sample (two global views)."""
        n = self.num_global_view
        counts = offset2bincount(point.offset).tolist()
        bs = len(counts) // n
        data = {}
        for key in ("feat", "coord", "origin_coord", "batch"):
            value = point[key].split(counts)
            value = list(chain(*[value[n * b : n * (b + 1)][::-1] for b in range(bs)]))
            if key == "batch":
                value = [torch.ones_like(v) * i for i, v in enumerate(value)]
            data[key] = torch.cat(value, dim=0)
        return Point(data)

    def up_cast(self, point):
        for _ in range(self.up_cast_level):
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point

    def distill_loss(self, target_sim, pred_sim, batch):
        loss = -torch.sum(target_sim * F.log_softmax(pred_sim / self.student_temp, dim=-1), dim=-1)
        return torch_scatter.segment_coo(loss, index=batch, reduce="mean").mean()

    @torch.no_grad()
    def diagnostics(self, target, pred_sim):
        """Collapse / learning checks: cross-entropy = target entropy + KL(target || student).
        KL should fall during training; target entropy and the number of prototypes the teacher
        uses should stay well above 0 (Sinkhorn keeps targets spread over prototypes)."""
        target = target.float()
        logp = F.log_softmax(pred_sim.float() / self.student_temp, dim=-1)
        t_ent = -(target * torch.log(target.clamp_min(1e-12))).sum(-1).mean()
        p_ent = -(logp.exp() * logp).sum(-1).mean()
        kl = (target * (torch.log(target.clamp_min(1e-12)) - logp)).sum(-1).mean()
        used = torch.unique(target.argmax(-1)).numel()
        agree = (target.argmax(-1) == logp.argmax(-1)).float().mean()
        # soft usage: perplexity of the batch-mean target (4096 = every prototype gets equal mass)
        mean_t = target.mean(0)
        soft_used = torch.exp(-(mean_t * torch.log(mean_t.clamp_min(1e-12))).sum())
        return dict(target_entropy=t_ent, pred_entropy=p_ent, mask_kl=kl,
                    protos_used=torch.tensor(float(used)), argmax_agree=agree, protos_soft_used=soft_used,
                    target_max=target.max(-1).values.mean(),
                    student_protos_used=torch.tensor(float(torch.unique(logp.argmax(-1)).numel())))

    # ---------------- forward ----------------
    def forward(self, data_dict):
        grid_size = self.grid_size
        with torch.no_grad():
            global_point = Point(
                feat=data_dict["global_feat"], coord=data_dict["global_coord"],
                origin_coord=data_dict["global_origin_coord"], offset=data_dict["global_offset"],
                grid_size=grid_size,
            )
            global_mask = self.generate_mask(global_point.coord, global_point.offset)
            mask_global_coord = global_point.coord.clone().detach()
            if self.mask_jitter is not None:
                mask_global_coord[global_mask] += torch.clip(
                    torch.randn_like(mask_global_coord[global_mask]).mul(self.mask_jitter),
                    max=self.mask_jitter * 2,
                )
            mask_global_point = Point(
                feat=data_dict["global_feat"], coord=mask_global_coord,
                origin_coord=data_dict["global_origin_coord"], mask=global_mask,
                offset=data_dict["global_offset"], grid_size=grid_size,
            )
            local_point = Point(
                feat=data_dict["local_feat"], coord=data_dict["local_coord"],
                origin_coord=data_dict["local_origin_coord"], offset=data_dict["local_offset"],
                grid_size=grid_size,
            )
            result = dict(loss=[])
            global_point_ = self.up_cast(self.teacher.backbone(global_point))
            global_feat = global_point_.feat

        # masked-to-global (aligned + rolled pairs)
        with torch.no_grad():
            mask_emb = self.teacher.mask_head.embed(global_feat)
            global_point_.feat = self.teacher.mask_head.prototype(mask_emb)
            mask_push = None
            mask_queue = self.queue_scores("mask", self.teacher.mask_head)
        mask_global_point_ = self.up_cast(self.student.backbone(mask_global_point))
        mask_pred_sim = self.student.mask_head(mask_global_point_.feat)
        if self.mask_loss_weight > 0:
            idx = self.match_neighbour(mask_global_point_.origin_coord, mask_global_point_.offset,
                                       global_point_.origin_coord, global_point_.offset)
            target = self.sinkhorn_knopp(global_point_.feat[idx[:, 1]], self.teacher_temp, extra=mask_queue)
            mask_push = (mask_emb[idx[:, 1]], mask_global_point_.batch[idx[:, 0]] // self.num_global_view)
            loss = self.distill_loss(target, mask_pred_sim[idx[:, 0]], mask_global_point_.batch[idx[:, 0]])
            result.update(self.diagnostics(target, mask_pred_sim[idx[:, 0]]))
            result["mask_loss"] = loss
            result["mask_match"] = torch.tensor(len(idx) / len(mask_global_point_.batch))
            result["loss"].append(loss * self.mask_loss_weight)
        if self.roll_mask_loss_weight > 0:
            roll_global_point_ = self.roll_point(global_point_)
            idx = self.match_neighbour(mask_global_point_.origin_coord, mask_global_point_.offset,
                                       roll_global_point_.origin_coord, roll_global_point_.offset)
            target = self.sinkhorn_knopp(roll_global_point_.feat[idx[:, 1]], self.teacher_temp, extra=mask_queue)
            loss = self.distill_loss(target, mask_pred_sim[idx[:, 0]], mask_global_point_.batch[idx[:, 0]])
            result["roll_mask_loss"] = loss
            result["roll_match"] = torch.tensor(len(idx) / len(mask_global_point_.batch))
            result["loss"].append(loss * self.roll_mask_loss_weight)

        # local-to-global (student local views vs teacher principal global view)
        if self.unmask_loss_weight > 0:
            with torch.no_grad():
                unmask_emb = self.teacher.unmask_head.embed(global_feat)
                global_point_.feat = self.teacher.unmask_head.prototype(unmask_emb)
                unmask_queue = self.queue_scores("unmask", self.teacher.unmask_head)
            local_point_ = self.up_cast(self.student.backbone(local_point))
            unmask_pred_sim = self.student.unmask_head(local_point_.feat)
            with torch.no_grad():
                principal = global_point_.batch % self.num_global_view == 0
                principal_batch = global_point_.batch[principal] // self.num_global_view
                idx = self.match_neighbour(
                    local_point_.origin_coord,
                    local_point_.offset[self.num_local_view - 1 :: self.num_local_view],
                    global_point_.origin_coord[principal],
                    batch2offset(principal_batch),
                )
                target = self.sinkhorn_knopp(global_point_.feat[principal][idx[:, 1]], self.teacher_temp,
                                             extra=unmask_queue)
                self.push_queue("unmask", unmask_emb[principal][idx[:, 1]],
                                local_point_.batch[idx[:, 0]] // self.num_local_view)
            loss = self.distill_loss(target, unmask_pred_sim[idx[:, 0]], local_point_.batch[idx[:, 0]])
            result["unmask_loss"] = loss
            result["unmask_match"] = torch.tensor(len(idx) / len(local_point_.batch))
            result["loss"].append(loss * self.unmask_loss_weight)

        if mask_push is not None:  # after use, so the current points are not counted twice
            self.push_queue("mask", *mask_push)
        if self.sinkhorn_cookies_now() is not None:
            result["sk_cookies"] = torch.tensor(float(self.sinkhorn_cookies_now()))
        result["loss"] = sum(result["loss"])
        result["mask_ratio"] = torch.tensor(self.mask_ratio)
        result["mask_size"] = torch.tensor(self.mask_size)
        result["teacher_temp"] = torch.tensor(self.teacher_temp)
        return result
