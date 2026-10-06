"""Profile streamed pre-training data (eagle_als.stream_pool) on one node, without a GPU.

Runs the pool loader for --minutes after the first batch and appends one JSON record to
<out>/results.jsonl: time to the first batch, samples/s, squares fetched (ok / errors / crashes),
cookies per fetched square, distinct squares per batch, and CPU cores used by the whole process
tree during the window. --rate caps consumption (samples/s) to imitate GPUs, so the measured CPU
and reuse are what training would see.

    python scripts/litePT/profile_pool.py --workers 32 --rate 40 --minutes 8 --out $OUT
"""

import argparse
import json
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import psutil

from eagle_als.train_utils import load_config


class CpuMeter:
    """Total CPU seconds of this process and all descendants (sampled every `every` s, so processes
    that exit between samples lose at most that much)."""

    def __init__(self, every=2.0):
        self.seen, self.every, self.stop = {}, every, False
        threading.Thread(target=self._run, daemon=True).start()

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
    """Terminate all descendants (DataLoader workers, fetch helpers) and wait for them."""
    procs = psutil.Process().children(recursive=True)
    for q in procs:
        try:
            q.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(procs, timeout=30)
    for q in alive:
        q.kill()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="scripts/litePT/configs/ssl_litept_s.py")
    p.add_argument("--opts", nargs="*", default=[])
    p.add_argument("--minutes", type=float, default=8.0, help="measurement window after the first batch")
    p.add_argument("--rate", type=float, default=0.0, help="cap consumption at this many samples/s (0: none)")
    p.add_argument("--batch-size", type=int, default=8, help="per-GPU batch")
    p.add_argument("--workers", type=int, default=16, help="DataLoader workers (each with a fetch helper)")
    p.add_argument("--pool-size", type=int, default=8, help="squares per worker")
    p.add_argument("--uses", type=int, default=24, help="cookies per square before retiring it")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    tag = a.tag or f"w{a.workers}_p{a.pool_size}_u{a.uses}_r{a.rate:g}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / f"{tag}_squares"
    cfg = load_config(a.config, a.opts)
    cfg.update(batch_size=a.batch_size, grad_accum=1, num_workers=a.workers, pool_size=a.pool_size,
               pool_uses_per_square=a.uses, pool_shm_dir=f"/dev/shm/eagle_profile_{os.getpid()}",
               pool_log_dir=str(log_dir), seed=a.seed)
    print(f"[profile] {tag} on {os.uname().nodename} ({len(os.sched_getaffinity(0))} cpus)", flush=True)

    from eagle_als.stream_pool import build_pool_loader

    meter = CpuMeter()
    _, batches = build_pool_loader(cfg, tag_square=True)
    t_start = time.time()
    it = iter(batches)
    rows = [next(it)]
    t0, cpu0 = time.time(), meter.total()
    times, n = [t0], 0
    while time.time() - t0 < a.minutes * 60:
        b = next(it)
        rows.append(b)
        times.append(time.time())
        n += b["num_samples"]
        if a.rate:
            time.sleep(max(0.0, t0 + n / a.rate - time.time()))
    t1, cpu1 = times[-1], meter.total()
    meter.stop = True
    kill_tree()
    shutil.rmtree(cfg["pool_shm_dir"], ignore_errors=True)

    recs = [json.loads(line) for f in log_dir.glob("*.jsonl") for line in open(f) if line.strip()]
    fetched = [r for r in recs if r["event"] == "fetched" and t0 <= r["t"] <= t1]
    retired = [r for r in recs if r["event"] == "retired"]
    status = {}
    for r in fetched:
        status[r["status"]] = status.get(r["status"], 0) + 1
    ok = status.get("ok", 0)
    rec = dict(
        tag=tag, cpus=len(os.sched_getaffinity(0)), workers=a.workers, pool_size=a.pool_size, uses_per_square=a.uses,
        rate_cap=a.rate, batch_size=a.batch_size, minutes=round((t1 - t0) / 60, 2),
        first_batch_s=round(t0 - t_start, 1), samples_per_s=round(n / (t1 - t0), 2),
        squares_ok_per_s=round(ok / (t1 - t0), 3), cookies_per_square=round(n / max(ok, 1), 1), square_status=status,
        retired_uses_median=float(np.median([r["uses"] for r in retired])) if retired else None,
        distinct_squares_per_batch=round(float(np.mean([len(set(b["square"])) / len(b["square"]) for b in rows])), 3),
        cores_used=round((cpu1 - cpu0) / (t1 - t0), 1),
    )
    print("[profile]", json.dumps(rec), flush=True)
    with open(out / "results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    os._exit(0)  # skip DataLoader teardown (its workers were already terminated)


if __name__ == "__main__":
    main()
