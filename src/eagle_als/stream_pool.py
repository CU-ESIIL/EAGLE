"""Streaming plan A: each DataLoader worker fetches its own 500 m squares and cuts cookies from them.

Alternative to the separate producer + shared buffer of eagle_als.stream (plan B); both share the
square sampler, fetch / thinning / block sorting (stream.produce_square) and cookie cutting
(stream.cut_cookie).

Per DataLoader worker:
    - a fetch helper process (`python -m eagle_als.stream_pool serve`, a plain subprocess because
      DataLoader workers are daemonic and cannot start multiprocessing children) draws square i from
      the seeded SquareSampler, fetches and thins it, and writes it as a shard to the worker's
      directory on /dev/shm. A PDAL crash only kills the helper, which is restarted.
    - a pool of `pool_size` squares. Each sample is a random 100 m cookie from a random pool square.
      A square is retired after `uses_per_square` cookies once a replacement has arrived; a late
      fetch never blocks (the worker keeps cutting from its pool and the logs show the real reuse).
Workers yield single samples (DataLoader batch_size=None, round-robin over workers), and
RoundRobinBatches collates consecutive samples, so a batch draws from several workers' pools.

    loader = build_pool_loader(cfg, rank, world)   # iterable of collated batches
"""

import argparse
import ctypes
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from litept.transform import Compose

from . import transforms  # noqa: F401  (registers ALS transforms)
from .stream import cut_cookie, produce_square, read_shard_meta

REC_PREFIX = "@@square "  # marks the helper's result lines on stdout (PDAL may print there too)


# ---------------- fetch helper (runs in its own process) ----------------
def serve(out_dir, seed=0, half_size=250.0, block=50.0, max_density=12.0, timeout=None, exclude_eval=True):
    """Read square indices from stdin, write each square as a shard into out_dir, print its record.

    Exits (removing out_dir) when stdin closes, i.e. when the owning worker exits."""
    from .stream import SquareSampler

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sampler = SquareSampler(seed=seed, **({} if exclude_eval else dict(eval_sites=None)))
    print(REC_PREFIX + json.dumps(dict(status="ready")), flush=True)
    for line in sys.stdin:
        i = int(line)
        name = f"{i:09d}"
        tile, lat, lon = sampler(i)
        rec = produce_square(out_dir, name, tile, lat, lon, half_size, block, max_density, 1000, timeout, seed)
        print(REC_PREFIX + json.dumps(rec), flush=True)
    shutil.rmtree(out_dir, ignore_errors=True)


def _die_with_parent():
    """preexec_fn: SIGTERM the helper when the worker that started it dies (Linux prctl)."""
    try:
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except OSError:
        pass


class FetchHelper:
    """A `serve` subprocess plus a reader thread that queues its result records."""

    def __init__(self, out_dir, **kw):
        self.out_dir, self.kw = Path(out_dir), kw
        self.restarts = -1
        self.start()

    def start(self):
        if self.restarts >= 20:
            raise RuntimeError(f"fetch helper for {self.out_dir} died {self.restarts} times")
        args = [sys.executable, "-m", "eagle_als.stream_pool", "serve", "--dir", str(self.out_dir)]
        for k, v in self.kw.items():
            if k == "exclude_eval":
                args += [] if v else ["--no-exclude-eval"]
            elif v is not None:
                args += [f"--{k.replace('_', '-')}", str(v)]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                                     preexec_fn=_die_with_parent)
        self.results = queue.Queue()
        self.ready = False
        self.inflight = deque()  # square indices sent and not yet answered (helper works FIFO)
        self.restarts += 1
        threading.Thread(target=self._read, args=(self.proc, self.results), daemon=True).start()

    @staticmethod
    def _read(proc, results):
        for line in proc.stdout:
            if line.startswith(REC_PREFIX):
                results.put(json.loads(line[len(REC_PREFIX):]))
        results.put(None)  # EOF: the helper exited

    def request(self, i):
        self.inflight.append(i)
        try:
            self.proc.stdin.write(f"{i}\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass  # the helper died; next() reports it

    def next(self, timeout=None):
        """Next record, None if nothing arrived within `timeout`, or a record with status "crashed"
        for the square the helper was working on when it died (the helper is then restarted and the
        other in-flight squares are re-requested)."""
        try:
            rec = self.results.get(timeout=timeout) if timeout else self.results.get_nowait()
        except queue.Empty:
            return None
        if rec is None:
            lost = list(self.inflight)
            self.proc.wait()
            self.start()
            for i in lost[1:]:
                self.request(i)
            return dict(name=f"{lost[0]:09d}", status="crashed", error="fetch helper died") if lost else None
        if rec.get("status") == "ready":
            self.ready = True
            return self.next(timeout)
        self.inflight.popleft()
        return rec

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.terminate()


# ---------------- dataset ----------------
class _Square:
    def __init__(self, rec, out_dir):
        self.rec = rec
        self.dir = Path(out_dir) / rec["name"]
        self.meta = read_shard_meta(self.dir)
        self.uses = 0
        self.arrival = time.time()


class SquarePoolDataset(IterableDataset):
    """Endless stream of transformed cookies cut from a per-worker pool of streamed squares (plan A).

    Square indices are split between all workers of all ranks (stream g = rank * W + worker takes
    squares g, g + G, g + 2G, ...), so runs are reproducible in which squares they draw, though not
    in the order samples arrive. With `log_dir`, each worker writes squares-r<rank>w<worker>.jsonl:
    one record per fetched square (status, timings) and per retired square (uses, life).
    """

    def __init__(self, transform, shm_root, pool_size=8, uses_per_square=24, min_pool=2, max_inflight=2,
                 radius=100.0, min_points=2000, half_size=250.0, block=50.0, max_density=12.0, seed=0,
                 rank=0, world=1, timeout=None, exclude_eval=True, log_dir=None, tag_square=False):
        self.transform = Compose(transform)
        self.shm_root = Path(shm_root)
        self.pool_size, self.uses_per_square, self.min_pool = pool_size, uses_per_square, min_pool
        self.max_inflight = max_inflight
        self.radius, self.min_points = radius, min_points
        self.helper_kw = dict(seed=seed, half_size=half_size, block=block, max_density=max_density,
                              timeout=timeout, exclude_eval=exclude_eval)
        self.seed, self.rank, self.world = seed, rank, world
        self.log_dir = None if log_dir is None else Path(log_dir)
        self.tag_square = tag_square

    def __iter__(self):
        wi = get_worker_info()
        w, n_workers = (wi.id, wi.num_workers) if wi else (0, 1)
        stream, n_streams = self.rank * n_workers + w, self.world * n_workers
        out_dir = self.shm_root / f"r{self.rank}w{w}"
        shutil.rmtree(out_dir, ignore_errors=True)
        rng = np.random.default_rng([self.seed, self.rank, w])
        log = None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log = open(self.log_dir / f"squares-r{self.rank}w{w}.jsonl", "a")
        helper = FetchHelper(out_dir, **self.helper_kw)
        pool, ready = [], deque()
        k = 0  # squares requested by this stream

        def record(**rec):
            if log is not None:
                log.write(json.dumps(dict(rec, t=time.time())) + "\n")
                log.flush()

        try:
            while True:
                # 1. keep the helper busy until the pool is full and one replacement is waiting
                want = self.pool_size - len(pool) + 1 - len(ready)
                while len(helper.inflight) < min(want, self.max_inflight):
                    helper.request(stream + k * n_streams)
                    k += 1
                # 2. collect finished fetches (block while the pool is too small to sample from)
                wait = len(pool) + len(ready) < self.min_pool
                rec = helper.next(timeout=5.0 if wait else None)
                while rec is not None:
                    record(event="fetched", **rec)
                    if rec["status"] == "ok":
                        ready.append(_Square(rec, out_dir))
                    rec = helper.next()
                # 3. fill the pool; retire the most-used square that reached its quota if one is waiting
                while ready and len(pool) < self.pool_size:
                    pool.append(ready.popleft())
                while ready:
                    done = [s for s in pool if s.uses >= self.uses_per_square]
                    if not done:
                        break
                    old = max(done, key=lambda s: s.uses)
                    pool.remove(old)
                    record(event="retired", name=old.rec["name"], uses=old.uses, life_s=time.time() - old.arrival)
                    shutil.rmtree(old.dir, ignore_errors=True)
                    pool.append(ready.popleft())
                if len(pool) < self.min_pool:
                    continue
                # 4. cut a cookie
                for _ in range(50):
                    sq = pool[rng.integers(len(pool))]
                    inner = sq.meta["half_size"] - self.radius
                    cx, cy = rng.uniform(-inner, inner, 2)
                    cookie = cut_cookie(sq.dir, cx, cy, self.radius, sq.meta)
                    sq.uses += 1  # empty cut (water, gap) counts too, so a mostly empty square retires
                    if len(cookie["xyz"]) >= self.min_points:
                        break
                else:
                    continue  # pool is mostly empty squares; try again after the next fetch
                sample = self.transform(cookie)
                if self.tag_square:
                    sample["square"] = f"r{self.rank}w{w}:{sq.rec['name']}"
                yield sample
        finally:
            helper.close()
            if log is not None:
                log.close()
            shutil.rmtree(out_dir, ignore_errors=True)


def _identity(x):
    return x


class RoundRobinBatches:
    """Collate every `batch_size` consecutive samples of a batch_size=None DataLoader into one batch.

    With in-order delivery the DataLoader takes samples from its workers in turn, so a batch spans
    min(batch_size, num_workers) workers' pools."""

    def __init__(self, loader, batch_size):
        self.loader, self.batch_size = loader, batch_size

    def __iter__(self):
        from .data import collate_points

        buf = []
        for sample in self.loader:
            buf.append(sample)
            if len(buf) == self.batch_size:
                yield collate_points(buf)
                buf = []


def build_pool_loader(cfg, rank=0, world=1, tag_square=False):
    """(dataset, iterable of per-GPU batches) for data_mode="pool" configs."""
    from .train_utils import worker_init_fn

    per_gpu = cfg["batch_size"] // world // cfg.get("grad_accum", 1)
    shm_root = Path(cfg.get("pool_shm_dir") or f"/dev/shm/eagle_pool_{os.getpid()}")
    ds = SquarePoolDataset(
        cfg["train_transform"], shm_root, pool_size=cfg.get("pool_size", 8),
        uses_per_square=cfg.get("pool_uses_per_square", 24), min_pool=cfg.get("pool_min", 2),
        min_points=cfg["min_points"], seed=cfg.get("seed", 0), rank=rank, world=world,
        log_dir=cfg.get("pool_log_dir"), tag_square=tag_square,
    )
    nw = cfg["num_workers"]
    loader = DataLoader(ds, batch_size=None, num_workers=nw, collate_fn=_identity, worker_init_fn=worker_init_fn,
                        prefetch_factor=cfg.get("pool_prefetch", 4) if nw else None,
                        multiprocessing_context="spawn" if nw else None)
    return ds, RoundRobinBatches(loader, per_gpu)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="fetch helper: square indices on stdin -> shards in --dir")
    s.add_argument("--dir", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--half-size", type=float, default=250.0)
    s.add_argument("--block", type=float, default=50.0)
    s.add_argument("--max-density", type=float, default=12.0)
    s.add_argument("--timeout", type=float, default=None)
    s.add_argument("--exclude-eval", dest="exclude_eval", action="store_true", default=True)
    s.add_argument("--no-exclude-eval", dest="exclude_eval", action="store_false")
    a = p.parse_args()
    serve(a.dir, a.seed, a.half_size, a.block, a.max_density, a.timeout, a.exclude_eval)


if __name__ == "__main__":
    main()
