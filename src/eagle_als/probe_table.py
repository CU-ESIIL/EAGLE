"""Multi-target probe task: several labels on one table of cookies, embedded once per evaluation.

Registered as `type="probe_table"` in `eval_tasks` (eagle_als.evaluation). One task = one labeled table
with a train/test split and a cookie cache; each target is probed on the rows where it has a label:

    dict(type="probe_table",
         name="lfrdb",                               # tensorboard: eval/lfrdb/*, eval_lfrdb/* (details)
         table="datasets/LFRDB_eval/lfrdb_eval.parquet",
         cache_dir=".../cache/lfrdb",                # built with scripts/litePT/build_eval_caches.py
         lat_col="lat", lon_col="lon",               # site location (cache building; id_col if missing)
         id_col="als_site_id",                       # cookie id; built from tile_col + lat/lon if absent
         tile_col="product_name_AWS",
         split_col="test_split",                     # "train" / "test" values, or bool (True = test)
         query=None,                                 # optional pandas row filter
         probes=("linear", "knn"),                   # optional, default both
         min_per_split=20,                           # classes / species need this many rows in train AND test
         targets=[
             dict(name="ecosys", kind="classification", col="ecosys"),
             dict(name="lifeform", kind="classification", col="ecosys_lifeform", drop=("N/A",)),
             dict(name="tree_cover", kind="regression", col="tree_cover_pct"),
             dict(name="ba", kind="regression", expr="basal_area_m2ha", transform="log1p"),
             dict(name="occ", kind="presence", cols="^occ__", strip="^occ__"),
         ])

Target kinds (rows with a missing label are left out of that target only):

- classification: `col` holds the class; values in `drop` are removed; classes with fewer than
  `min_per_split` rows in either split are removed. Logistic regression and cosine kNN, as in the
  `classification` task (eagle_als.probe): <target>_<probe>_bal_acc, <target>_<probe>_f1, per-class
  F1 in eval_<task>/<target>_<probe>_f1/<class>, and <target>_chance.
- regression: `col`, or `expr` (pandas eval on the table, e.g. "trees_s1 / plot_area_ha"), optionally
  `transform="log1p"` (fitted and scored on the transformed scale). Ridge regression on standardized
  embeddings (alpha by leave-one-out CV) and cosine kNN regression: <target>_<probe>_r2 on the test rows.
- presence: one binary label per column; `cols` is a list or a regex on the column names, value > 0 =
  present (`strip`: optional regex removed from column names in metric tags). A column is kept if it
  has at least `min_per_split` presences and absences in both splits.
  One class-balanced logistic regression per column (C = 1, the sklearn default, fitted jointly) and
  cosine kNN (similarity-weighted share of present neighbours). Headline <target>_<probe>_auc (mean
  ROC AUC over columns) and <target>_<probe>_ap (mean average precision), per column
  eval_<task>/<target>_<probe>_auc/<column>, and <target>_n (number of columns kept).
"""

import re
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .cache import cookie_path
from .evaluation import EvalTask, register
from .probe import REPO_ROOT, embed, prepare_samples, probe_scores
from .sites import add_site_ids, read_table


def tag(name):
    return re.sub(r"[^\w.-]+", "_", str(name)).strip("_")


def load_task_table(spec):
    """The task's table with `id_col` (built from tile + lat/lon when the table has no such column)."""
    table = read_table(REPO_ROOT / spec["table"])  # an absolute path stays as it is
    if "geometry" in table:
        table = table.drop(columns="geometry")
    id_col = spec.get("id_col", "als_site_id")
    if id_col not in table:
        if not spec.get("lat_col"):
            raise ValueError(f"eval task {spec['name']}: no column {id_col!r} and no lat_col/lon_col to build it")
        table = add_site_ids(table, spec["lat_col"], spec["lon_col"], spec.get("tile_col", "product_name_AWS"))
        if id_col != "als_site_id":
            table = table.rename(columns={"als_site_id": id_col})
    return table


def split_mask(series):
    return (series.astype(bool) if series.dtype == bool else series.astype(str).str.lower() == "test").to_numpy()


def _target_values(table, t):
    if "expr" in t:
        return table.eval(t["expr"])
    return table[t["col"]]


@register("probe_table")
class TableProbe(EvalTask):
    """One labeled table, its cookies preprocessed once; run() embeds them and probes every target."""

    def __init__(self, spec, cfg):
        super().__init__(spec, cfg)
        table = load_task_table(spec)
        id_col, split_col = spec.get("id_col", "als_site_id"), spec["split_col"]
        self.probes = tuple(spec.get("probes", ("linear", "knn")))
        self.knn_k = spec.get("knn_k", cfg.get("eval_knn_k", 20))
        min_split = spec.get("min_per_split", 20)
        n0 = len(table)
        if spec.get("query"):
            table = table.query(spec["query"])
        table = table[table[id_col].notna() & table[split_col].notna()].copy()
        table["_path"] = [str(cookie_path(spec["cache_dir"], s)) for s in table[id_col]]
        table = table[table["_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
        is_test = split_mask(table[split_col])
        # a cookie shared by train and test rows (e.g. two BBS routes with the same start point) would leak
        both = table.groupby(id_col)[split_col].transform(lambda s: len(set(split_mask(s))) > 1).to_numpy(bool)
        if both.any():
            print(f"[eval] {self.name}: dropped {both.sum()} rows whose cookie is in both splits", flush=True)
            table, is_test = table[~both].reset_index(drop=True), is_test[~both]

        self.targets, used = [], np.zeros(len(table), bool)
        for t in spec["targets"]:
            kind, name = t["kind"], t["name"]
            if kind == "classification":
                y = _target_values(table, t)
                ok = y.notna() & ~y.isin(t.get("drop", ()))
                n_tr = y[ok & ~is_test].value_counts()
                n_te = y[ok & is_test].value_counts()
                keep = [c for c in n_tr.index if n_tr[c] >= min_split and n_te.get(c, 0) >= min_split]
                ok &= y.isin(keep)
                rows = np.where(ok)[0]
                classes = sorted(keep)
                if len(classes) < 2:
                    print(f"[eval] {self.name}/{name}: fewer than 2 classes with >= {min_split} rows per split, "
                          f"skipped", flush=True)
                    continue
                labels = y.iloc[rows].map({c: i for i, c in enumerate(classes)}).to_numpy()
                extra = dict(class_tags=[tag(c) for c in classes])
                desc = f"{len(classes)} classes (of {y[y.notna()].nunique()})"
            elif kind == "regression":
                y = _target_values(table, t).astype(float)
                rows = np.where(np.isfinite(y.to_numpy()))[0]
                labels = y.to_numpy()[rows]
                if t.get("transform") == "log1p":
                    labels = np.log1p(labels)
                elif t.get("transform"):
                    raise ValueError(f"eval task {self.name}/{name}: unknown transform {t['transform']!r}")
                extra, desc = {}, "regression" + (f" ({t['transform']})" if t.get("transform") else "")
            elif kind == "presence":
                cols = t["cols"]
                cols = [c for c in table.columns if re.search(cols, c)] if isinstance(cols, str) else list(cols)
                v = table[cols].to_numpy(dtype=float)
                known = np.isfinite(v)
                pres = (v > 0) & known
                absn = (v <= 0) & known
                good = np.ones(len(cols), bool)
                for m in (~is_test, is_test):
                    good &= (pres[m].sum(0) >= min_split) & (absn[m].sum(0) >= min_split)
                cols = [c for c, g in zip(cols, good) if g]
                if not cols:
                    print(f"[eval] {self.name}/{name}: no column with >= {min_split} presences and absences per "
                          f"split, skipped", flush=True)
                    continue
                v = v[:, good]
                rows = np.where(np.isfinite(v).any(1))[0]
                labels = np.where(np.isfinite(v[rows]), (v[rows] > 0).astype(float), np.nan)
                extra = dict(col_tags=[tag(re.sub(t["strip"], "", c) if t.get("strip") else c) for c in cols])
                desc = f"{len(cols)} columns (of {len(good)})"
            else:
                raise ValueError(f"eval task {self.name}/{name}: unknown target kind {kind!r}")
            used[rows] = True
            self.targets.append(dict(name=name, kind=kind, rows=rows, y=labels, is_test=is_test[rows], **extra))
            print(f"[eval] {self.name}/{name}: {desc}, {(~is_test[rows]).sum()} train / {is_test[rows].sum()} test",
                  flush=True)
        if not self.targets:
            raise ValueError(f"eval task {self.name}: no usable target (cache_dir {spec['cache_dir']})")

        # embed each cookie once, even if several rows share it
        paths = table["_path"].to_numpy()
        uniq, inverse = np.unique(paths[used], return_inverse=True)
        self.row_cookie = np.full(len(table), -1)
        self.row_cookie[np.where(used)[0]] = inverse
        self.samples = prepare_samples(uniq.tolist(), cfg)
        print(f"[eval] {self.name}: {used.sum()} rows with a cached cookie and a usable label ({n0} in table), "
              f"{len(uniq)} cookies, {np.mean([len(s['coord']) for s in self.samples]):.0f} voxels per cookie",
              flush=True)

    def run(self, model, device):
        x_all = embed(model, self.samples, device)
        out = {}
        for t in self.targets:
            x = x_all[self.row_cookie[t["rows"]]]
            if t["kind"] == "classification":
                scores = probe_scores(x, t["y"], t["is_test"], probes=self.probes, knn_k=self.knn_k,
                                      class_tags=t["class_tags"])
            elif t["kind"] == "regression":
                scores = regression_scores(x, t["y"], t["is_test"], probes=self.probes, knn_k=self.knn_k)
            else:
                scores = presence_scores(x, t["y"], t["is_test"], probes=self.probes, knn_k=self.knn_k,
                                         col_tags=t["col_tags"])
            out.update({f"{t['name']}_{k}": v for k, v in scores.items()})
        return out


def _cosine_neighbours(train_x, test_x, k):
    a = F.normalize(torch.from_numpy(train_x).float(), dim=1)
    b = F.normalize(torch.from_numpy(test_x).float(), dim=1)
    sim, idx = (b @ a.T).topk(min(k, len(a)), dim=1)
    w = sim.clamp(min=0) + 1e-6
    return (w / w.sum(1, keepdim=True)).numpy(), idx.numpy()


def regression_scores(x, y, is_test, probes=("linear", "knn"), knn_k=20):
    """R2 on the test rows of ridge regression (standardized inputs, LOO-chosen alpha) and kNN regression."""
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr, te = ~is_test, is_test
    out = {}
    if "linear" in probes:
        reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13)))
        reg.fit(x[tr], y[tr])
        out["linear_r2"] = r2_score(y[te], reg.predict(x[te]))
    if "knn" in probes:
        w, idx = _cosine_neighbours(x[tr], x[te], knn_k)
        out["knn_r2"] = r2_score(y[te], (w * y[tr][idx]).sum(1))
    return out


def fit_presence_logistic(x, y, C=1.0, max_iter=200):
    """Class-balanced L2 logistic regression for every column of y (0/1, NaN = unknown) at once.

    Minimizes, per column, C * sum_i w_i * logloss_i + ||coef||^2 / 2 with sklearn's "balanced" weights,
    i.e. the same model as LogisticRegression(class_weight="balanced") fitted column by column.
    Returns a predict function: test x -> scores (n, columns).
    """
    mu, sd = x.mean(0), x.std(0) + 1e-6
    X = torch.from_numpy((x - mu) / sd).double()
    Y = torch.from_numpy(np.nan_to_num(y)).double()
    known = torch.from_numpy(np.isfinite(y)).double()
    n, n_pos = known.sum(0), (Y * known).sum(0)
    wts = known * (Y * (n / (2 * n_pos)) + (1 - Y) * (n / (2 * (n - n_pos))))
    W = torch.zeros(X.shape[1], Y.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(Y.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-6)

    def closure():
        opt.zero_grad()
        loss = C * (wts * F.binary_cross_entropy_with_logits(X @ W + b, Y, reduction="none")).sum() \
            + 0.5 * (W ** 2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    W, b = W.detach(), b.detach()
    return lambda xt: (torch.from_numpy((xt - mu) / sd).double() @ W + b).numpy()


def presence_scores(x, y, is_test, probes=("linear", "knn"), knn_k=20, col_tags=None):
    """Mean ROC AUC and average precision over columns of y, plus per-column AUC."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    tr, te = ~is_test, is_test
    col_tags = col_tags or [str(i) for i in range(y.shape[1])]
    out = dict(n=y.shape[1])
    for name in probes:
        if name == "linear":
            score = fit_presence_logistic(x[tr], y[tr])(x[te])
        else:
            w, idx = _cosine_neighbours(x[tr], x[te], knn_k)
            ytr = y[tr]
            score = np.stack([(w * np.nan_to_num(ytr[:, j][idx], nan=0.0)).sum(1)
                              / np.maximum((w * np.isfinite(ytr[:, j][idx])).sum(1), 1e-12)
                              for j in range(y.shape[1])], 1)
        aucs, aps = [], []
        for j in range(y.shape[1]):
            m = np.isfinite(y[te, j])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aucs.append(roc_auc_score(y[te, j][m], score[m, j]))
                aps.append(average_precision_score(y[te, j][m], score[m, j]))
            out[f"{name}_auc/{col_tags[j]}"] = aucs[-1]
        out[f"{name}_auc"] = float(np.mean(aucs))
        out[f"{name}_ap"] = float(np.mean(aps))
    return out
