"""Linear probes on frozen embeddings, scored on each task's held-out test rows.

`evaluate(task, X)` fits on the train rows and scores the test rows; X has one row per `task.df` row
(rows with a NaN embedding are left out and counted). Returns (summary dict, per-target DataFrame).

    multiclass   standardize + multinomial logistic regression (class-balanced); C chosen by 3-fold CV on train.
                 Only classes with >= MIN_TRAIN train rows and >= MIN_TEST_CLASS test rows are scored/predicted
                 (or >= min_count in both, when min_count is given).
                 Reported: balanced accuracy, macro F1, accuracy, chance (1 / n classes).
    multilabel   one logistic regression per species (class-balanced); one C per task, chosen by 3-fold CV on
                 train over the most common species. Species with >= min_count positives in train and in test
                 (default 20). Reported: mean AUROC and mean average precision
                 (with mean prevalence as the AP of a random ranking).
    regression   standardize + ridge, alpha by RidgeCV on train. Per target: R2, Spearman, RMSE on the test
                 rows with a measured value.

min_count sets the class-set scenario, so one full table gives both "all classes" (min_count=1) and
">= 20 per split" (min_count=20) results; None keeps the legacy per-kind defaults. Regression ignores it.

Hyperparameter selection uses only train rows (random folds, so slightly optimistic inside train; the test
split is spatial and untouched).
"""

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

MIN_TRAIN = 20  # multiclass: rows per class in train
MIN_TEST_CLASS = 5  # multiclass: rows per class in test
C_GRID = (0.01, 0.1, 1.0)
ALPHAS = np.logspace(-2, 4, 13)


def _scale(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _logreg(C):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=C, class_weight="balanced", max_iter=1000)


def _fold_score(C, X, y, fit_idx, val_idx, scoring):
    """Score of one CV fold (what cross_val_score computes per fold)."""
    from sklearn.metrics import get_scorer

    return get_scorer(scoring)(_logreg(C).fit(X[fit_idx], y[fit_idx]), X[val_idx], y[val_idx])


def _cv_grid(X, ys, cv, scoring, n_jobs):
    """Mean CV score per C in C_GRID, averaged over folds and then over the targets in `ys`.

    All (C, target, fold) fits run as one flat parallel batch: the same folds and scorer as calling
    cross_val_score per C and target, but parallel over everything instead of 3 folds at a time.
    """
    jobs = [(C, y, a, b) for C in C_GRID for y in ys for a, b in cv.split(X, y)]
    s = Parallel(n_jobs=n_jobs)(delayed(_fold_score)(C, X, y, a, b, scoring) for C, y, a, b in jobs)
    return np.asarray(s).reshape(len(C_GRID), len(ys), -1).mean(2).mean(1)


def evaluate(task, X, n_jobs=-1, seed=0, min_count=None):
    ok = np.isfinite(X).all(1)
    tr, te = ok & ~task.df["is_test"].to_numpy(), ok & task.df["is_test"].to_numpy()
    fn = dict(multiclass=_multiclass, multilabel=_multilabel, regression=_regression)[task.kind]
    summary, per = fn(task, X, tr, te, n_jobs, seed, min_count)
    summary.update(task=task.name, min_count=min_count, kind=task.kind, n_train=int(tr.sum()), n_test=int(te.sum()),
                   n_no_embedding=int((~ok).sum()), n_dim=X.shape[1])
    per.insert(0, "task", task.name)
    return summary, per


def _multiclass(task, X, tr, te, n_jobs, seed, min_count):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold

    y = task.y["label"].to_numpy()
    n_tr, n_te = pd.Series(y[tr]).value_counts(), pd.Series(y[te]).value_counts()
    min_tr, min_te = (MIN_TRAIN, MIN_TEST_CLASS) if min_count is None else (min_count, min_count)
    classes = sorted(c for c in n_tr.index if n_tr[c] >= min_tr and n_te.get(c, 0) >= min_te)
    if len(classes) < 2:
        return dict(n_classes=len(classes)), pd.DataFrame()
    tr, te = tr & np.isin(y, classes), te & np.isin(y, classes)
    Xtr, Xte = _scale(X[tr], X[te])
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    scores = _cv_grid(Xtr, [y[tr]], cv, "balanced_accuracy", n_jobs)
    C = C_GRID[int(np.argmax(scores))]
    pred = _logreg(C).fit(Xtr, y[tr]).predict(Xte)
    f1 = f1_score(y[te], pred, labels=classes, average=None, zero_division=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bal = balanced_accuracy_score(y[te], pred)
    summary = dict(n_classes=len(classes), n_classes_total=int(len(n_tr)), C=C, chance=1 / len(classes),
                   bal_acc=bal, macro_f1=float(f1.mean()), acc=accuracy_score(y[te], pred),
                   test_rows_scored_frac=float(te.sum() / max(1, (task.df["is_test"].to_numpy() & np.isfinite(X).all(1)).sum())))
    per = pd.DataFrame({"target": classes, "n_train": [n_tr[c] for c in classes],
                        "n_test": [n_te[c] for c in classes], "f1": f1})
    return summary, per


def _fit_species(Xtr, ytr, Xte, yte, C):
    from sklearn.metrics import average_precision_score, roc_auc_score

    p = _logreg(C).fit(Xtr, ytr).decision_function(Xte)
    return roc_auc_score(yte, p), average_precision_score(yte, p)


def _multilabel(task, X, tr, te, n_jobs, seed, min_count):
    from sklearn.model_selection import StratifiedKFold

    Y = task.y.to_numpy()
    min_tr = min_te = 20 if min_count is None else min_count
    pos_tr, pos_te = Y[tr].sum(0), Y[te].sum(0)
    keep = np.where((pos_tr >= min_tr) & (pos_te >= min_te) & (pos_tr < tr.sum()) & (pos_te < te.sum()))[0]
    if len(keep) == 0:
        return dict(n_species=0), pd.DataFrame()
    Xtr, Xte = _scale(X[tr], X[te])
    Ytr, Yte = Y[tr], Y[te]
    common = keep[np.argsort(-pos_tr[keep])[: min(15, len(keep))]]  # C is chosen on the most common species
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    scores = _cv_grid(Xtr, [Ytr[:, j] for j in common], cv, "roc_auc", n_jobs)
    C = C_GRID[int(np.argmax(scores))]
    res = Parallel(n_jobs=n_jobs)(delayed(_fit_species)(Xtr, Ytr[:, j], Xte, Yte[:, j], C) for j in keep)
    auc, ap = np.array(res).T
    prev = Yte[:, keep].mean(0)
    summary = dict(n_species=len(keep), n_species_total=Y.shape[1], C=C, mean_auroc=float(auc.mean()),
                   mean_ap=float(ap.mean()), mean_prevalence=float(prev.mean()))
    per = pd.DataFrame({"target": task.y.columns[keep], "n_train": pos_tr[keep], "n_test": pos_te[keep],
                        "auroc": auc, "ap": ap, "prevalence_test": prev})
    return summary, per


def _regression(task, X, tr, te, n_jobs, seed, min_count):
    from scipy.stats import spearmanr
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score

    rows = []
    for col in task.y.columns:
        y = task.y[col].to_numpy(float)
        m_tr, m_te = tr & np.isfinite(y), te & np.isfinite(y)
        if m_tr.sum() < 10 or m_te.sum() < 5:
            continue
        Xtr, Xte = _scale(X[m_tr], X[m_te])
        model = RidgeCV(alphas=ALPHAS).fit(Xtr, y[m_tr])
        pred = model.predict(Xte)
        rows.append(dict(target=col, n_train=int(m_tr.sum()), n_test=int(m_te.sum()), alpha=model.alpha_,
                         r2=r2_score(y[m_te], pred), spearman=spearmanr(y[m_te], pred)[0],
                         rmse=float(np.sqrt(np.mean((y[m_te] - pred) ** 2))), test_std=float(y[m_te].std())))
    per = pd.DataFrame(rows)
    summary = dict(n_targets=len(per), mean_r2=float(per["r2"].mean()), mean_spearman=float(per["spearman"].mean()))
    return summary, per
