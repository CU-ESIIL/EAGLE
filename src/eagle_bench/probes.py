"""Linear probes on frozen embeddings, scored on each task's held-out test rows.

`evaluate(task, X)` fits on the train rows and scores the test rows; X has one row per `task.df` row
(rows with a NaN embedding are left out and counted). Returns (summary dict, per-target DataFrame).

Every probe is the project's single linear-probe implementation, eagle_als.linear_probe: one torch.nn.Linear
layer on standardized features, full-batch L-BFGS, class-balanced, L2 penalty chosen by 3-fold CV on the train
rows (see that module for the exact recipe). One fit per task covers all of its classes / species / targets.

    multiclass   softmax over the scored classes. Only classes with >= MIN_TRAIN train rows and >= MIN_TEST_CLASS
                 test rows are scored/predicted (or >= min_count in both, when min_count is given).
                 Reported: balanced accuracy, macro F1, accuracy, chance (1 / n classes).
    multilabel   one sigmoid output per species. Species with >= min_count positives in train and in test
                 (default 20). Reported: mean AUROC and mean average precision
                 (with mean prevalence as the AP of a random ranking).
    regression   one output per target (rows without a value are masked per target).
                 Per target: R2, Spearman, RMSE on the test rows with a measured value.

min_count sets the class-set scenario, so one full table gives both "all classes" (min_count=1) and
">= 20 per split" (min_count=20) results; None keeps the legacy per-kind defaults. Regression ignores it.

Hyperparameter selection uses only train rows (random folds, so slightly optimistic inside train; the test
split is spatial and untouched).
"""

import warnings

import numpy as np
import pandas as pd

from eagle_als import linear_probe as lp

MIN_TRAIN = 20  # multiclass: rows per class in train
MIN_TEST_CLASS = 5  # multiclass: rows per class in test


def evaluate(task, X, n_jobs=-1, seed=0, min_count=None, device="auto"):
    """n_jobs: CPU threads for torch (-1 = leave torch's default)."""
    import torch

    if n_jobs and n_jobs > 0:
        torch.set_num_threads(n_jobs)
    X = np.asarray(X, dtype=np.float32)
    ok = np.isfinite(X).all(1)
    tr, te = ok & ~task.df["is_test"].to_numpy(), ok & task.df["is_test"].to_numpy()
    fn = dict(multiclass=_multiclass, multilabel=_multilabel, regression=_regression)[task.kind]
    summary, per = fn(task, X, tr, te, dict(seed=seed, device=device), min_count)
    summary.update(task=task.name, min_count=min_count, kind=task.kind, n_train=int(tr.sum()), n_test=int(te.sum()),
                   n_no_embedding=int((~ok).sum()), n_dim=X.shape[1])
    per.insert(0, "task", task.name)
    return summary, per


def _multiclass(task, X, tr, te, kw, min_count):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    y = task.y["label"].to_numpy()
    n_tr, n_te = pd.Series(y[tr]).value_counts(), pd.Series(y[te]).value_counts()
    min_tr, min_te = (MIN_TRAIN, MIN_TEST_CLASS) if min_count is None else (min_count, min_count)
    classes = sorted(c for c in n_tr.index if n_tr[c] >= min_tr and n_te.get(c, 0) >= min_te)
    if len(classes) < 2:
        return dict(n_classes=len(classes)), pd.DataFrame()
    tr, te = tr & np.isin(y, classes), te & np.isin(y, classes)
    idx = {c: i for i, c in enumerate(classes)}
    probe = lp.fit_select(X[tr], np.array([idx[c] for c in y[tr]]), "multiclass", n_classes=len(classes), **kw)
    pred = np.asarray(classes)[probe.predict(X[te])]
    yte = y[te]
    f1 = f1_score(yte, pred, labels=classes, average=None, zero_division=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bal = balanced_accuracy_score(yte, pred)
    summary = dict(n_classes=len(classes), n_classes_total=int(len(n_tr)), l2=probe.l2, chance=1 / len(classes),
                   bal_acc=bal, macro_f1=float(f1.mean()), acc=accuracy_score(yte, pred),
                   test_rows_scored_frac=float(te.sum() / max(1, (task.df["is_test"].to_numpy() & np.isfinite(X).all(1)).sum())))
    per = pd.DataFrame({"target": classes, "n_train": [n_tr[c] for c in classes],
                        "n_test": [n_te[c] for c in classes], "f1": f1})
    return summary, per


def _multilabel(task, X, tr, te, kw, min_count):
    from sklearn.metrics import average_precision_score, roc_auc_score

    Y = task.y.to_numpy().astype(float)
    min_tr = min_te = 20 if min_count is None else min_count
    pos_tr, pos_te = Y[tr].sum(0), Y[te].sum(0)
    keep = np.where((pos_tr >= min_tr) & (pos_te >= min_te) & (pos_tr < tr.sum()) & (pos_te < te.sum()))[0]
    if len(keep) == 0:
        return dict(n_species=0), pd.DataFrame()
    Ytr, Yte = Y[tr][:, keep], Y[te][:, keep]
    probe = lp.fit_select(X[tr], Ytr, "multilabel", **kw)
    s = probe.scores(X[te])
    auc = np.array([roc_auc_score(Yte[:, j], s[:, j]) for j in range(len(keep))])
    ap = np.array([average_precision_score(Yte[:, j], s[:, j]) for j in range(len(keep))])
    prev = Yte.mean(0)
    summary = dict(n_species=len(keep), n_species_total=Y.shape[1], l2=probe.l2, mean_auroc=float(auc.mean()),
                   mean_ap=float(ap.mean()), mean_prevalence=float(prev.mean()))
    per = pd.DataFrame({"target": task.y.columns[keep], "n_train": pos_tr[keep], "n_test": pos_te[keep],
                        "auroc": auc, "ap": ap, "prevalence_test": prev})
    return summary, per


def _regression(task, X, tr, te, kw, min_count):
    from scipy.stats import spearmanr

    Y = task.y.to_numpy(float)
    known = np.isfinite(Y)
    cols = [j for j in range(Y.shape[1]) if (tr & known[:, j]).sum() >= 10 and (te & known[:, j]).sum() >= 5]
    if not cols:
        return dict(n_targets=0), pd.DataFrame()
    Y = Y[:, cols]
    probe = lp.fit_select(X[tr], Y[tr], "regression", **kw)
    P = probe.scores(X[te])
    rows = []
    for k, j in enumerate(cols):
        m = np.isfinite(Y[te][:, k])
        y, p = Y[te][m, k], P[m, k]
        rows.append(dict(target=task.y.columns[j], n_train=int((tr & known[:, j]).sum()), n_test=int(m.sum()),
                         l2=probe.l2, r2=1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                         spearman=spearmanr(y, p)[0], rmse=float(np.sqrt(np.mean((y - p) ** 2))),
                         test_std=float(y.std())))
    per = pd.DataFrame(rows)
    summary = dict(n_targets=len(per), l2=probe.l2, mean_r2=float(per["r2"].mean()),
                   mean_spearman=float(per["spearman"].mean()))
    return summary, per
