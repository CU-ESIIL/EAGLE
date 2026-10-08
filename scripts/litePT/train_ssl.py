"""Self-supervised pre-training of LitePT on 3DEP cookies (ForPT / Sonata recipe, src/eagle_als/ssl.py).

Data (config `data_mode`): "pool" streams 500 m squares from 3DEP inside the DataLoader workers
(eagle_als.stream_pool; the full-node workflow is slurm/pretrain.sbatch); "cache" reads cookies
cached with eagle_als.cache (used by the overfit test).

Single GPU:
    python scripts/litePT/train_ssl.py --config scripts/litePT/configs/ssl_litept_s.py
Multi-GPU (one node):
    torchrun --standalone --nproc-per-node 4 scripts/litePT/train_ssl.py --config ... --opts batch_size=64
grad_accum and num_workers default to "auto" (train_utils.resolve_batching): the global batch_size stays
the same on any number of GPUs, with at most max_batch_per_gpu samples per GPU per micro-batch.

The run resumes automatically from <out_dir>/checkpoints/latest.txt, so the same command can be
resubmitted to continue a run that hit its slurm time limit. Logs go to <out_dir>/log.txt and
tensorboard (<out_dir>/tb; the Text tab holds scripts/litePT/TENSORBOARD.md, how to read the curves).
Exported encoder weights: <out_dir>/backbone_teacher.pth. With `eval_tasks` in the config, rank 0 runs
validation tasks on the frozen teacher encoder every `eval_every` steps, e.g. NLCD land-cover probes
(eagle_als.evaluation, eagle_als.probe; tensorboard eval/* and eval_<task>/*).
"""

import argparse
import json
import os

# fewer fragmented-memory OOMs: the Sinkhorn matrices change size every step (several GB with the memory bank)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from eagle_als.data import CookiePoolDataset, collate_points
from eagle_als.ssl import SonataLitePT
from eagle_als.train_utils import (
    InfiniteRandomSampler, StopFlag, all_reduce_mean, amp_dtype, is_main, latest_checkpoint, layerwise_param_groups,
    load_config, lr_factor, move_to, resolve_batching, save_checkpoint, seed_everything, setup_distributed,
    worker_init_fn,
)


def build_loader(cfg, rank, world, epoch=0, start_step=0):
    """(dataset, iterable of per-GPU batches)."""
    if cfg.get("data_mode", "pool") == "pool":
        from eagle_als.stream_pool import build_pool_loader

        return build_pool_loader(cfg, rank, world, seed_offset=start_step)
    ds = CookiePoolDataset(cfg["cache_dirs"], cfg["train_transform"], min_points=cfg["min_points"],
                           max_samples=cfg.get("max_samples"))
    sampler = InfiniteRandomSampler(ds, seed=cfg["seed"] + epoch, rank=rank, world=world)
    per_gpu = cfg["batch_size"] // world // cfg["grad_accum"]
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
    per_gpu, accum = resolve_batching(cfg, world)  # grad_accum / num_workers "auto" -> this job's GPUs and CPUs
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(cfg["seed"] + rank)
    out_dir = Path(cfg["out_dir"])
    cfg["pool_log_dir"] = cfg.get("pool_log_dir") or str(out_dir / "pool")  # per-worker square logs
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
              f"batch={cfg['batch_size']} = {world} GPUs x {per_gpu} x accum {accum}, "
              f"{cfg['num_workers']} workers/GPU out={out_dir}", flush=True)
        guide = Path(__file__).with_name("TENSORBOARD.md")
        if guide.exists():
            writer.add_text("guide", guide.read_text(), step)

    eval_tasks = []
    if is_main() and cfg.get("eval_tasks"):
        from eagle_als.evaluation import build_tasks

        eval_tasks = build_tasks(cfg["eval_tasks"], cfg)
    eval_every = cfg.get("eval_every", 0) if cfg.get("eval_tasks") else 0

    def evaluate():
        """Validation tasks on rank 0, encoder frozen; the other ranks wait (keep it well under the NCCL
        timeout, 10 min)."""
        if is_main():
            from eagle_als.evaluation import run_tasks, tb_tag

            t0 = time.time()
            logs = run_tasks(model, eval_tasks, device)
            for k, v in logs.items():
                writer.add_scalar(tb_tag(k), v, step)
            headline = {k: v for k, v in logs.items() if tb_tag(k).startswith("eval/")}
            print(f"[eval] step {step} ({time.time() - t0:.0f} s) "
                  + " ".join(f"{k}={v:.3f}" for k, v in headline.items()), flush=True)
            log_file.write(json.dumps(dict(step=step, eval=logs)) + "\n")
            log_file.flush()
        if world > 1:
            dist.barrier()

    stop = StopFlag(cfg.get("max_hours"))
    total = cfg["total_steps"]
    steps_per_epoch = cfg.get("steps_per_epoch", 2000)  # optimizer steps between checkpoints of `epoch`
    streamed = cfg.get("data_mode", "pool") == "pool"
    model.train()
    if eval_every and step == 0:
        evaluate()  # random-initialisation baseline
    mem_max = 0.0
    it = None
    while step < total:
        if it is None or not streamed:  # cached mode re-scans the cache each epoch; streaming keeps its workers
            ds, loader = build_loader(cfg, rank, world, epoch, start_step=step)
            it = iter(loader)
        if is_main():
            pool = "streaming squares from 3DEP" if streamed else f"pool of {len(ds)} cookies"
            print(f"[ssl] epoch {epoch}: {pool}", flush=True)
        t_data, t_last = 0.0, time.time()
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
                    # mem_gb: peak since the previous log line; mem_max_gb: peak since the job started
                    mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
                    mem_max = max(mem_max, mem)
                    logs.update(lr=opt.param_groups[-1]["lr"], grad_norm=float(gnorm), momentum=momentum,
                                n_points=int(batch["global_coord"].shape[0]),
                                step_s=dt / cfg["log_every"], data_frac=t_data / max(dt, 1e-6),
                                mem_gb=mem, mem_max_gb=mem_max)
                    for k, v in logs.items():
                        writer.add_scalar(f"train/{k}", v, step)
                    print(f"[ssl] step {step}/{total} " + " ".join(f"{k}={v:.4g}" for k, v in logs.items()), flush=True)
                    log_file.write(json.dumps(dict(step=step, **logs)) + "\n")
                    log_file.flush()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                t_data, t_last = 0.0, time.time()
            if eval_every and step % eval_every == 0:
                evaluate()
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
