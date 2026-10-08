"""Classification eval task: probes on frozen teacher embeddings of fixed labeled cookies.

Registered as `type="classification"` in `eval_tasks` (eagle_als.evaluation). A task is a table of
cached cookies with one class label each:

    dict(type="classification",
         name="nlcd",                                # tensorboard: eval/nlcd/*, eval_nlcd/* (per class)
         table="datasets/NLCD_eval/nlcd_lidar_eval.parquet",  # csv / parquet; relative to the repo root
         cache_dir=".../cache/nlcd",                 # cookie cache built with python -m eagle_als.cache
         label_col="nlcd_class",
         id_col="als_site_id",                       # cookie id in the cache (default als_site_id)
         split_col="test_split",                     # optional: "train" / "test" values, or bool (True =
                                                     #   test); without it, stratified 5-fold CV
         query="balanced_subset",                    # optional: pandas query selecting rows
         probes=("linear", "knn"),                   # optional: which probes to fit (default both)
         knn_k=20,                                   # optional (default: config eval_knn_k)
         min_per_class=5)                            # optional: drop classes with fewer usable rows

Each cookie is turned into one deterministic view (fixed 8 pts/m2, central `view_radius` disc, no
augmentation) once, when the run starts, and kept in memory, so every evaluation sees the same inputs.
The teacher encoder embeds each cookie as the mean of its up-cast point features (stage 2 + 3 + 4,
the features the SSL heads see). The probes fitted on those embeddings:

- linear: logistic regression on standardized embeddings (linear separability)
- knn: cosine kNN, k = `knn_k` (local neighbourhood structure, no fitting)

Reported per probe: balanced accuracy and macro F1 (headline), F1 of every class in the test rows
(`<probe>_f1/<class>`), plus the chance level (1 / number of classes).
"""

import random
import re
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from litept.model import Point
from litept.transform import Compose

from . import transforms  # noqa: F401  (registers ALS transforms)
from .cache import cookie_path
from .data import collate_points
from .evaluation import EvalTask, register
from .fetch import load_cookie
from .sites import read_table
from .train_utils import move_to


def eval_transform(cfg, density=8.0):
    """One deterministic-ish view per cookie: fixed density, central disc, no augmentation."""
    return Compose([
        dict(type="ALSFeatures"),
        dict(type="RandomPointThinning", target_density=(density, density), radius=100.0, p=1.0),
        dict(type="XYCrop", radius=cfg["view_radius"]),
        dict(type="CenterShift", apply_z=False),
        dict(type="GridSample", grid_size=cfg["grid_size"], hash_type="fnv", mode="train"),
        dict(type="PackFeat", keys=tuple(cfg["feat_keys"])),
    ])


def prepare_samples(paths, cfg):
    """Fixed model inputs for a list of cookie files (seeded per cookie; global RNG state is restored)."""
    tf = eval_transform(cfg)
    py_state, np_state = random.getstate(), np.random.get_state()
    samples = []
    try:
        for i, path in enumerate(paths):
            random.seed(i)
            np.random.seed(i)
            samples.append(tf(load_cookie(path)[0]))
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    return samples


REPO_ROOT = Path(__file__).resolve().parents[2]


@register("classification")
class ClassificationProbe(EvalTask):
    """One labeled cookie set, preprocessed into fixed model inputs; run() embeds it and fits the probes."""

    def __init__(self, spec, cfg):
        super().__init__(spec, cfg)
        table = read_table(REPO_ROOT / spec["table"])  # an absolute path stays as it is
        id_col, label_col, split_col = spec.get("id_col", "als_site_id"), spec["label_col"], spec.get("split_col")
        min_per_class = spec.get("min_per_class", 5)
        self.probes = tuple(spec.get("probes", ("linear", "knn")))
        if set(self.probes) - {"linear", "knn"}:
            raise ValueError(f"eval task {self.name}: probes must be 'linear' and/or 'knn', got {self.probes}")
        self.knn_k = spec.get("knn_k", cfg.get("eval_knn_k", 20))
        n0 = len(table)
        if spec.get("query"):
            table = table.query(spec["query"])
        n_query = len(table)
        table = table[table[id_col].notna() & table[label_col].notna()].copy()
        table["_path"] = [str(cookie_path(spec["cache_dir"], s)) for s in table[id_col]]
        table = table[table["_path"].map(lambda p: Path(p).exists())]
        counts = table[label_col].value_counts()
        dropped = sorted(counts[counts < min_per_class].index)
        table = table[table[label_col].isin(counts[counts >= min_per_class].index)].reset_index(drop=True)
        if table[label_col].nunique() < 2:
            raise ValueError(f"eval task {self.name}: fewer than 2 classes with >= {min_per_class} cached cookies "
                             f"(cache_dir {spec['cache_dir']})")
        self.classes = sorted(table[label_col].unique())
        self.class_tags = [re.sub(r"[^\w.-]+", "_", str(c)).strip("_") for c in self.classes]
        self.labels = table[label_col].map({c: i for i, c in enumerate(self.classes)}).to_numpy()
        self.is_test = None
        if split_col:
            split = table[split_col]
            self.is_test = (split.astype(bool) if split.dtype == bool else split.astype(str).str.lower() == "test")
            self.is_test = self.is_test.to_numpy()
        self.samples = prepare_samples(table["_path"].tolist(), cfg)
        split_txt = (f"{(~self.is_test).sum()} train / {self.is_test.sum()} test" if self.is_test is not None
                     else "5-fold CV")
        print(f"[eval] {self.name}: {len(table)} rows usable of {n_query} selected ({n0} in table; needs a cached "
              f"cookie and a class with >= {min_per_class} rows), {len(self.classes)} classes, {split_txt}, "
              f"{np.mean([len(s['coord']) for s in self.samples]):.0f} voxels per cookie"
              + (f"; dropped classes {dropped}" if dropped else ""), flush=True)

    def run(self, model, device):
        x = embed(model, self.samples, device)
        return probe_scores(x, self.labels, self.is_test, probes=self.probes, knn_k=self.knn_k,
                            class_tags=self.class_tags)


@torch.no_grad()
def embed(model, samples, device, batch_size=8):
    """Mean-pooled up-cast teacher features, one row per cookie (float32, on CPU)."""
    out = []
    for i in range(0, len(samples), batch_size):
        b = move_to(collate_points(samples[i : i + batch_size]), device)
        point = Point(feat=b["feat"], coord=b["coord"], offset=b["offset"], grid_size=model.grid_size)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            point = model.up_cast(model.teacher.backbone(point))
        feat, batch = point.feat.float(), point.batch
        sums = torch.zeros(len(b["offset"]), feat.shape[1], device=device).index_add_(0, batch, feat)
        out.append((sums / torch.bincount(batch, minlength=len(b["offset"])).clamp(min=1)[:, None]).cpu())
    return torch.cat(out).numpy()


def knn_predict(train_x, train_y, test_x, k, n_classes):
    """Cosine kNN with similarity-weighted votes."""
    a = F.normalize(torch.from_numpy(train_x).float(), dim=1)
    b = F.normalize(torch.from_numpy(test_x).float(), dim=1)
    sim, idx = (b @ a.T).topk(min(k, len(a)), dim=1)
    votes = torch.zeros(len(b), n_classes).scatter_add_(1, torch.from_numpy(train_y)[idx], sim.clamp(min=0) + 1e-6)
    return votes.argmax(1).numpy()


def probe_scores(x, y, is_test=None, probes=("linear", "knn"), knn_k=20, n_folds=5, seed=0, class_tags=None):
    """Balanced accuracy, macro F1 and per-class F1 of each probe (held-out test rows, or stratified CV).

    Macro F1 averages over the classes present in the test rows; a class with no test rows gets no
    per-class score. Per-class keys are "<probe>_f1/<class tag>".
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_classes = int(y.max()) + 1
    class_tags = class_tags or [str(i) for i in range(n_classes)]
    if is_test is not None:
        folds = [(np.where(~is_test)[0], np.where(is_test)[0])]
    else:
        folds = StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(x, y)
    preds, truth = {p: [] for p in probes}, []
    for tr, te in folds:
        if "linear" in probes:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
            clf.fit(x[tr], y[tr])
            preds["linear"].append(clf.predict(x[te]))
        if "knn" in probes:
            preds["knn"].append(knn_predict(x[tr], y[tr], x[te], knn_k, n_classes))
        truth.append(y[te])
    truth = np.concatenate(truth)
    present = np.unique(truth)
    out = dict(chance=1.0 / n_classes)
    for name in probes:
        pred = np.concatenate(preds[name])
        with warnings.catch_warnings():  # predictions of classes with no test rows are expected
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            out[f"{name}_bal_acc"] = balanced_accuracy_score(truth, pred)
        per_class = f1_score(truth, pred, labels=present, average=None, zero_division=0)
        out[f"{name}_f1"] = float(per_class.mean())
        for c, f in zip(present, per_class):
            out[f"{name}_f1/{class_tags[c]}"] = f
    return out
