"""Datasets and collate functions for cached ALS cookies (see eagle_als.cache)."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from litept.transform import Compose

from . import transforms  # noqa: F401  (registers ALS transforms)
from .cache import cookie_path, read_manifest
from .fetch import load_cookie


def pool_paths(cache_dirs, min_points=2000):
    """Paths of usable cookies in one or more cache directories (status ok, >= min_points)."""
    if isinstance(cache_dirs, (str, Path)):
        cache_dirs = [cache_dirs]
    paths = []
    for c in cache_dirs:
        m = read_manifest(c)
        if len(m) == 0:
            continue
        m = m[(m.status == "ok") & (m.n_points >= min_points)]
        paths += [str(cookie_path(c, s)) for s in m.site_id]
    return [p for p in paths if Path(p).exists()]


class CookiePoolDataset(Dataset):
    """Unlabeled cookies for self-supervised pre-training.

    The pool is the set of cookies cached so far; call `refresh()` (the trainer does this every
    epoch when `refresh_pool=True`) to pick up cookies that a concurrent cache job has added.
    """

    def __init__(self, cache_dirs, transform, min_points=2000, max_samples=None):
        self.cache_dirs = cache_dirs
        self.min_points = min_points
        self.max_samples = max_samples
        self.transform = Compose(transform)
        self.refresh()

    def refresh(self):
        self.paths = pool_paths(self.cache_dirs, self.min_points)
        if self.max_samples:
            self.paths = self.paths[: self.max_samples]
        if not self.paths:
            raise RuntimeError(f"no cached cookies found in {self.cache_dirs}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            cookie, _ = load_cookie(self.paths[idx])
        except Exception as e:  # corrupted file: fall back to a random other cookie
            print(f"[data] failed to read {self.paths[idx]}: {e}", flush=True)
            return self[np.random.randint(len(self))]
        return self.transform(cookie)


class StreamingChunkDataset(Dataset):
    """Unlabeled cookies cut at random from the shard buffer of a running producer (eagle_als.stream).

    Read-only. Each item ignores its index: it picks a random shard, a random center in the square's
    inner part (so the whole `radius` cookie lies inside the square) and cuts the cookie from the
    blocks it overlaps. The shard list is re-read every `relist_every` s. Cookies with fewer than
    `min_points` points (water, data gaps, square partly outside the tile) are redrawn. With
    `log_dir`, each worker appends the shard of every sample to usage-<pid>.txt.
    """

    def __init__(self, buffer_dir, transform, radius=100.0, min_points=2000, relist_every=5.0,
                 virtual_len=10**9, log_dir=None, wait_timeout=1800):
        self.buffer_dir = Path(buffer_dir)
        self.transform = Compose(transform)
        self.radius = radius
        self.min_points = min_points
        self.relist_every = relist_every
        self.virtual_len = virtual_len
        self.log_dir = log_dir
        self.wait_timeout = wait_timeout
        self._shards, self._listed, self._usage = [], 0.0, []

    def __len__(self):
        return self.virtual_len

    def _shard_list(self):
        import time

        from .stream import list_shards

        t0 = time.time()
        while time.time() - self._listed > self.relist_every or not self._shards:
            self._shards, self._listed = list_shards(self.buffer_dir), time.time()
            if self._shards:
                break
            if time.time() - t0 > self.wait_timeout:
                raise RuntimeError(f"no shards in {self.buffer_dir} after {self.wait_timeout}s")
            time.sleep(2.0)
        return self._shards

    def sample_cookie(self, max_tries=20):
        from .stream import cut_cookie, read_shard_meta

        for _ in range(max_tries):
            shards = self._shard_list()
            name = shards[np.random.randint(len(shards))]
            try:
                meta = read_shard_meta(self.buffer_dir / name)
                inner = meta["half_size"] - self.radius
                cx, cy = np.random.uniform(-inner, inner, 2)
                cookie = cut_cookie(self.buffer_dir / name, cx, cy, self.radius, meta)
            except FileNotFoundError:  # evicted between listing and opening
                self._listed = 0.0
                continue
            if len(cookie["xyz"]) >= self.min_points:
                return cookie, name
        raise RuntimeError(f"no cookie with >= {self.min_points} points after {max_tries} tries")

    def _log_usage(self, name):
        if self.log_dir is None:
            return
        self._usage.append(name)
        if len(self._usage) >= 50:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(self.log_dir) / f"usage-{os.getpid()}.txt", "a") as f:
                f.write("\n".join(self._usage) + "\n")
            self._usage = []

    def __getitem__(self, idx):
        cookie, name = self.sample_cookie()
        self._log_usage(name)
        return self.transform(cookie)


def build_ssl_dataset(cfg):
    """Pre-training dataset from a config: cached cookie pool (default) or streamed shards
    (data_mode="stream", stream_dir=<producer buffer>)."""
    if cfg.get("data_mode", "cache") == "stream":
        return StreamingChunkDataset(cfg["stream_dir"], cfg["train_transform"], min_points=cfg["min_points"],
                                     log_dir=cfg.get("stream_usage_dir"))
    return CookiePoolDataset(cfg["cache_dirs"], cfg["train_transform"], min_points=cfg["min_points"],
                             max_samples=cfg.get("max_samples"))


class LabeledCookieDataset(Dataset):
    """Labeled cookies for downstream classification.

    table: rows with `als_site_id` (from check_lidar_sources.py) and a label column.
    classes: ordered list of class names (label -> index).
    """

    def __init__(self, table, label_col, classes, cache_dir, transform, loop=1):
        self.table = table.reset_index(drop=True)
        self.label_col = label_col
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.paths = [str(cookie_path(cache_dir, s)) for s in self.table.als_site_id]
        self.labels = np.array([self.class_to_idx[c] for c in self.table[label_col]], dtype=np.int64)
        self.transform = Compose(transform)
        self.loop = loop

    def __len__(self):
        return len(self.table) * self.loop

    def __getitem__(self, idx):
        idx = idx % len(self.table)
        cookie, _ = load_cookie(self.paths[idx])
        d = self.transform(cookie)
        d["label"] = np.array([self.labels[idx]])
        d["index"] = np.array([idx])
        return d


def collate_points(batch):
    """Concatenate a list of sample dicts into one batch of LitePT inputs.

    - keys `<prefix>_offset` / `offset` hold cumulative point counts per view; they are shifted so
      offsets index into the concatenated batch (as LitePT expects).
    - a sample without explicit offset gets offset = [len(coord)].
    - per-point arrays are concatenated; per-sample arrays (label, index) are concatenated too.
    """
    batch = [dict(b) for b in batch]
    for b in batch:
        if "coord" in b and "offset" not in b:
            b["offset"] = np.array([len(b["coord"])])
        b.pop("index_valid_keys", None)
    out = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if key.endswith("offset"):
            shift, shifted = 0, []
            for v in vals:
                shifted.append(np.asarray(v) + shift)
                shift += int(np.asarray(v)[-1])
            out[key] = torch.from_numpy(np.concatenate(shifted)).long()
        elif isinstance(vals[0], np.ndarray):
            arr = np.concatenate(vals, axis=0)
            out[key] = torch.from_numpy(arr)
            if arr.dtype.kind in "iu":
                out[key] = out[key].long()
        elif isinstance(vals[0], torch.Tensor):
            out[key] = torch.cat(vals)
        else:
            out[key] = vals
    out["num_samples"] = len(batch)
    return out
