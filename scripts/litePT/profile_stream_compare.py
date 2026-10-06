"""Compare streaming plan A (eagle_als.stream_pool) and plan B (eagle_als.stream + StreamingChunkDataset).

Runs one design for --minutes on this node and appends one JSON record to <out>/results.jsonl:
time to the first batch, samples/s, squares fetched (ok / errors / crashes), reuse (cookies per
fetched square), distinct squares per batch, and CPU cores used by the whole process tree.
--rate caps consumption (samples/s) to imitate GPUs, so spare CPU goes to fetching.

    python scripts/litePT/profile_stream_compare.py --design A --workers 32 --minutes 8 --out $OUT
    python scripts/litePT/profile_stream_compare.py --design B --producer-workers 32 --workers 16 --out $OUT
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

from eagle_als.data import StreamingChunkDataset, collate_points
from eagle_als.train_utils import load_config, worker_init_fn


class TaggedStreamDataset(StreamingChunkDataset):
    """Plan B dataset that also returns the shard name of each sample."""

    def __getitem__(self, idx):
        cookie, name = self.sample_cookie()
        sample = self.transform(cookie)
        sample["square"] = name
        return sample


class CpuMeter:
    """Total CPU seconds of this process and all descendants (sampled every `every` s, so processes
    that exit between samples lose at most that much)."""

    def __init__(self, every=2.0):
        self.seen, self.every, self.stop = {}, every, False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        me = psutil.Process()
        while not self.stop:
            for p in [me] + me.children(recursive=True):
                try:
                    t = p.cpu_times()
                    self.seen[p.pid] = t.user + t.system + t.children_user + t.children_system
                except psutil.Error:
                    pass
            time.sleep(self.every)

    def total(self):
        return sum(self.seen.values())


def kill_tree():
    """Terminate all descendants (DataLoader workers, fetch helpers, producer) and wait for them."""
    procs = psutil.Process().children(recursive=True)
    for q in procs:
        try:
            q.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(procs, timeout=30)
    for q in alive:
        q.kill()


def jsonl(path):
    try:
        return [json.loads(line) for line in open(path) if line.strip()]
    except FileNotFoundError:
        return []


def consume(batches, minutes, rate, on_first=None):
    """Iterate batches for `minutes` after the first one, optionally rate-limited; per-batch log."""
    rows, it = [], iter(batches)
    first = next(it)
    t0 = time.time()
    if on_first is not None:
        on_first()
    n = first["num_samples"]
    rows.append(dict(t=t0, n=first["num_samples"], squares=first["square"]))
    while time.time() - t0 < minutes * 60:
        b = next(it)
        n += b["num_samples"]
        rows.append(dict(t=time.time(), n=b["num_samples"], squares=b["square"]))
        if rate:
            time.sleep(max(0.0, t0 + n / rate - time.time()))
    return rows


def run_a(a, cfg, out, meter):
    from eagle_als.stream_pool import build_pool_loader

    log_dir = out / f"{a.tag}_squares"
    cfg.update(num_workers=a.workers, pool_size=a.pool_size, pool_uses_per_square=a.uses,
               pool_shm_dir=f"/dev/shm/eagle_cmp_{os.getpid()}", pool_log_dir=str(log_dir), seed=a.seed)
    _, batches = build_pool_loader(cfg, tag_square=True)
    t_start = time.time()
    cpu_first = {}
    rows = consume(batches, a.minutes, a.rate, on_first=lambda: cpu_first.update(s=meter.total()))
    extra = dict(cpu_s=meter.total(), cpu_first_s=cpu_first["s"])
    kill_tree()
    recs = [r for p in log_dir.glob("*.jsonl") for r in jsonl(p)]
    fetched = [r for r in recs if r["event"] == "fetched"]
    retired = [r for r in recs if r["event"] == "retired"]
    extra.update(retired_uses_median=float(np.median([r["uses"] for r in retired])) if retired else None,
                 retired_life_s_median=float(np.median([r["life_s"] for r in retired])) if retired else None)
    shutil.rmtree(cfg["pool_shm_dir"], ignore_errors=True)
    return t_start, rows, fetched, extra


def run_b(a, cfg, out, meter):
    import torch
    from torch.utils.data import DataLoader

    buf = Path(a.buffer or os.environ.get("LOCAL", "/tmp")) / f"eagle_cmp_buffer_{os.getpid()}"
    state = out / f"{a.tag}_producer"
    shutil.rmtree(buf, ignore_errors=True)
    shutil.rmtree(state, ignore_errors=True)
    t_start = time.time()
    prod = subprocess.Popen([sys.executable, "-m", "eagle_als.stream", "--buffer", str(buf), "--state-dir", str(state),
                             "--workers", str(a.producer_workers), "--max-shards", str(a.max_shards),
                             "--seed", str(a.seed)])
    from eagle_als.stream import list_shards

    while len(list_shards(buf)) < a.min_shards:  # the trainer's start-up wait
        if prod.poll() is not None:
            raise RuntimeError("producer exited")
        time.sleep(1.0)
    t_buffer = time.time() - t_start
    ds = TaggedStreamDataset(buf, cfg["train_transform"], min_points=cfg["min_points"])
    loader = DataLoader(ds, batch_size=a.batch_size, sampler=range(10**9), num_workers=a.workers,
                        collate_fn=collate_points, worker_init_fn=worker_init_fn, prefetch_factor=4,
                        multiprocessing_context="spawn")
    try:
        cpu_first = {}
        rows = consume(loader, a.minutes, a.rate, on_first=lambda: cpu_first.update(s=meter.total()))
        cpu_s = meter.total()
    finally:
        prod.send_signal(15)  # producer flushes its log on SIGTERM
        try:
            prod.wait(timeout=120)
        except subprocess.TimeoutExpired:
            prod.kill()
        kill_tree()
        shutil.rmtree(buf, ignore_errors=True)
    fetched = [dict(r, t=r["arrival"]) for r in jsonl(state / "producer.jsonl")]
    del torch
    return t_start, rows, fetched, dict(cpu_s=cpu_s, cpu_first_s=cpu_first["s"], buffer_fill_s=round(t_buffer, 1), min_shards=a.min_shards)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--design", choices=["A", "B"], required=True)
    p.add_argument("--config", default="scripts/litePT/configs/ssl_litept_s.py")
    p.add_argument("--opts", nargs="*", default=[])
    p.add_argument("--minutes", type=float, default=8.0, help="measurement window after the first batch")
    p.add_argument("--rate", type=float, default=0.0, help="cap consumption at this many samples/s (0: none)")
    p.add_argument("--batch-size", type=int, default=8, help="per-GPU batch")
    p.add_argument("--workers", type=int, default=16, help="DataLoader workers (A: also one fetch helper each)")
    p.add_argument("--pool-size", type=int, default=8, help="A: squares per worker")
    p.add_argument("--uses", type=int, default=24, help="A: cookies per square before retiring it")
    p.add_argument("--producer-workers", type=int, default=32, help="B: producer fetch processes")
    p.add_argument("--min-shards", type=int, default=50, help="B: wait for this many shards before training")
    p.add_argument("--max-shards", type=int, default=2000, help="B: buffer size")
    p.add_argument("--buffer", default=None, help="B: buffer root (default $LOCAL)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    a.tag = a.tag or f"{a.design}_w{a.workers}" + (f"_p{a.producer_workers}" if a.design == "B" else "") + f"_r{a.rate:g}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.config, a.opts)
    cfg["batch_size"], cfg["grad_accum"] = a.batch_size, 1
    cpus = len(os.sched_getaffinity(0))
    print(f"[compare] {a.tag} on {os.uname().nodename} ({cpus} cpus)", flush=True)

    meter = CpuMeter()
    cpu0 = meter.total()
    t_start, rows, fetched, extra = (run_a if a.design == "A" else run_b)(a, cfg, out, meter)
    t0, t1 = rows[0]["t"], rows[-1]["t"]
    cpu = extra.pop("cpu_s") - cpu0
    cpu_window = cpu + cpu0 - extra.pop("cpu_first_s")
    meter.stop = True

    n = sum(r["n"] for r in rows[1:])
    win = [r for r in fetched if t0 <= r["t"] <= t1]
    ok = sum(r["status"] == "ok" for r in win)
    status = {}
    for r in win:
        status[r["status"]] = status.get(r["status"], 0) + 1
    distinct = [len(set(r["squares"])) / len(r["squares"]) for r in rows]
    rec = dict(
        tag=a.tag, design=a.design, cpus=cpus, workers=a.workers, rate_cap=a.rate, batch_size=a.batch_size,
        minutes=round((t1 - t0) / 60, 2), first_batch_s=round(t0 - t_start, 1),
        samples_per_s=round(n / (t1 - t0), 2), squares_ok_per_s=round(ok / (t1 - t0), 3),
        cookies_per_square=round(n / max(ok, 1), 1), square_status=status,
        distinct_squares_per_batch=round(float(np.mean(distinct)), 3),
        cores_used=round(cpu / (t1 - t_start), 1), cores_used_window=round(cpu_window / (t1 - t0), 1), **extra,
        **({"pool_size": a.pool_size, "uses_per_square": a.uses} if a.design == "A" else
           {"producer_workers": a.producer_workers}),
    )
    print("[compare]", json.dumps(rec), flush=True)
    with open(out / "results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    os._exit(0)  # do not wait for daemon threads / DataLoader teardown


if __name__ == "__main__":
    main()
