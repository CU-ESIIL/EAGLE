"""Self-supervised pre-training of LitePT on cached 3DEP cookies (ForPT / Sonata recipe, src/eagle_als/ssl.py).

Single GPU:
    python scripts/litePT/train_ssl.py --config scripts/litePT/configs/ssl_litept_s.py
Multi-GPU (one node):
    torchrun --standalone --nproc-per-node 4 scripts/litePT/train_ssl.py --config ... --opts batch_size=64

The run resumes automatically from <out_dir>/checkpoints/latest.txt, so the same command can be
resubmitted to continue a run that hit its slurm time limit. Logs go to <out_dir>/log.txt and
tensorboard (<out_dir>/tb). Exported encoder weights: <out_dir>/backbone_teacher.pth.
"""

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from eagle_als.data import build_ssl_dataset, collate_points
from eagle_als.ssl import SonataLitePT
from eagle_als.train_utils import (
    InfiniteRandomSampler, StopFlag, all_reduce_mean, amp_dtype, is_main, latest_checkpoint, layerwise_param_groups,
    load_config, lr_factor, move_to, save_checkpoint, seed_everything, setup_distributed, worker_init_fn,
)


def build_loader(cfg, rank, world, epoch=0):
    ds = build_ssl_dataset(cfg)
    sampler = InfiniteRandomSampler(ds, seed=cfg["seed"] + epoch, rank=rank, world=world)
    per_gpu = cfg["batch_size"] // world // cfg.get("grad_accum", 1)
    assert per_gpu * world * cfg.get("grad_accum", 1) == cfg["batch_size"], "batch_size must divide evenly" 
    loader = DataLoader(
        ds, batch_size=per_gpu, sampler=sampler, num_workers=cfg["num_workers"], collate_fn=collate_points,
        pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn,
        persistent_workers=False, prefetch_factor=cfg["prefetch_factor"] if cfg["num_workers"] > 0 else None,
        multiprocessing_context="spawn" if cfg["num_workers"] > 0 else None,
    )
    return ds, loader


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--opts", nargs="*", default=[], help="config overrides key=value")
    a = p.parse_args()
    cfg = load_config(a.config, a.opts)
    rank, local_rank, world = setup_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(cfg["seed"] + rank)
    out_dir = Path(cfg["out_dir"])
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    model = SonataLitePT(cfg["backbone"], **cfg["ssl"]).to(device)
    lr = cfg["lr"]
    groups = layerwise_param_groups(model.student, lr, cfg["weight_decay"], cfg["backbone"]["enc_depths"],
                                    cfg["layer_decay"])
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=cfg["weight_decay"])
    dtype = amp_dtype(cfg["amp_dtype"])
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16 and device.type == "cuda"))
    ddp_model = (DDP(model, device_ids=[local_rank], find_unused_parameters=cfg.get("find_unused_parameters", False))
                 if world > 1 else model)
    accum = cfg.get("grad_accum", 1)

    step, epoch = 0, 0
    ckpt = latest_checkpoint(out_dir)
    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step, epoch = state["step"], state.get("epoch", 0) + 1
        if is_main():
            print(f"[ssl] resumed from {ckpt} at step {step}", flush=True)

    writer = None
    if is_main():
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(out_dir / "tb")
        log_file = open(out_dir / "log.txt", "a")
        n_params = sum(p.numel() for p in model.student.backbone.parameters()) / 1e6
        print(f"[ssl] world={world} amp={dtype} backbone params={n_params:.1f}M lr={lr:.2e} "
              f"batch={cfg['batch_size']} (accum {accum}) out={out_dir}", flush=True)

    stop = StopFlag(cfg.get("max_hours"))
    total = cfg["total_steps"]
    steps_per_epoch = cfg.get("steps_per_epoch", 2000)  # optimizer steps between pool refreshes
    model.train()
    while step < total:
        ds, loader = build_loader(cfg, rank, world, epoch)
        if is_main():
            pool = f"streamed from {cfg['stream_dir']}" if cfg.get("data_mode") == "stream" else f"pool of {len(ds)} cookies"
            print(f"[ssl] epoch {epoch}: {pool}", flush=True)
        t_data, t_last = 0.0, time.time()
        it = iter(loader)
        for i in range(steps_per_epoch):
            model.set_step(step, total)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_factor(step, total, cfg["warmup_steps"], cfg["lr_schedule"])
            opt.zero_grad(set_to_none=True)
            for micro in range(accum):
                t0 = time.time()
                batch = move_to(next(it), device)
                t_data += time.time() - t0
                sync = micro == accum - 1 or world == 1
                ctx = nullcontext() if sync else ddp_model.no_sync()
                with ctx:
                    with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                        losses = ddp_model(batch)
                    scaler.scale(losses["loss"] / accum).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.student.parameters(), cfg["clip_grad"])
            scaler.step(opt)
            scaler.update()
            momentum = model.update_teacher()
            if cfg.get("empty_cache"):
                torch.cuda.empty_cache()
            step += 1

            if step % cfg["log_every"] == 0:
                logs = {k: all_reduce_mean(v.detach().float().to(device)).item() for k, v in losses.items()}
                if is_main():
                    dt = time.time() - t_last
                    logs.update(lr=opt.param_groups[-1]["lr"], grad_norm=float(gnorm), momentum=momentum,
                                n_points=int(batch["global_coord"].shape[0]),
                                step_s=dt / cfg["log_every"], data_frac=t_data / max(dt, 1e-6),
                                mem_gb=torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0)
                    for k, v in logs.items():
                        writer.add_scalar(f"train/{k}", v, step)
                    print(f"[ssl] step {step}/{total} " + " ".join(f"{k}={v:.4g}" for k, v in logs.items()), flush=True)
                    log_file.write(json.dumps(dict(step=step, **logs)) + "\n")
                    log_file.flush()
                t_data, t_last = 0.0, time.time()
            stopping = stop.should_stop() if step % 10 == 0 else False
            if is_main() and (step % cfg["ckpt_every"] == 0 or step >= total or stopping):
                state = dict(model=model.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                             step=step, epoch=epoch, cfg=cfg)
                path = save_checkpoint(out_dir, state, step, keep=cfg["keep_ckpts"])
                torch.save(model.backbone_state_dict("teacher"), out_dir / "backbone_teacher.pth")
                torch.save(model.backbone_state_dict("student"), out_dir / "backbone_student.pth")
                print(f"[ssl] saved {path}", flush=True)
            if stopping:
                if is_main():
                    print("[ssl] stop requested (time limit / signal); exiting after checkpoint", flush=True)
                return
            if step >= total:
                break
        epoch += 1
    if is_main():
        print("[ssl] training complete", flush=True)


if __name__ == "__main__":
    main()
