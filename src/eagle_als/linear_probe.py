"""Linear probe: one `torch.nn.Linear` layer, the single probing implementation for every EAGLE experiment.

Used by the benchmark (eagle_bench.probes) and by the evaluation during pre-training (eagle_als.probe,
eagle_als.probe_table), so every probe result in the project comes from the same model and training recipe:

    inputs       standardized with the training rows' mean and SD (constant features left at 0)
    model        y_hat = x @ W + b, one output per class / species / regression target
    loss         multiclass  softmax cross-entropy, class-balanced sample weights n / (K * n_class)
                 multilabel  sigmoid BCE per target, class-balanced per target (n / 2n_pos, n / 2n_neg)
                 regression  0.5 * squared error per target, targets standardized with the training rows
                 Missing labels (NaN) are masked. Each target's loss is averaged over its labelled rows, and
                 the per-target losses are summed, so targets do not compete: the joint fit equals separate
                 fits of each target with the same l2.
    penalty      l2 / 2 * ||W||^2 (bias not penalized)
    optimizer    full-batch L-BFGS (strong Wolfe line search) from zero weights: deterministic and
                 convex, so no learning rate, epochs or seeds to tune
    l2           chosen from L2_GRID by K-fold CV on the training rows (fit_select), scored with
                 balanced accuracy / mean AUROC / mean R2; warm-started along the grid, ties go to the
                 stronger penalty

Run on GPU when one is visible (device="auto"), else CPU; set torch.set_num_threads to the CPUs you have.
"""

import warnings

import numpy as np
import torch
import torch.nn.functional as F

# Penalty on the per-row mean loss, strongest first (warm-start path, tie-break). The strong end matters for small
# tables with many features (e.g. ~100 OFO plots x 1024-d: an l2 of 0.1 still overfits badly).
L2_GRID = (1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
MAX_ITER = 500
KINDS = ("multiclass", "multilabel", "regression")


def _device(device):
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class LinearProbe:
    """A fitted probe. scores(X): logits (multiclass, multilabel) or predictions in target units (regression)."""

    def __init__(self, kind, linear, mu, sd, y_mu=None, y_sd=None):
        self.kind, self.linear, self.mu, self.sd, self.y_mu, self.y_sd = kind, linear, mu, sd, y_mu, y_sd
        self.l2 = None
        self.cv_scores = None

    @torch.no_grad()
    def scores(self, X):
        p = next(self.linear.parameters())
        Xs = (torch.as_tensor(np.asarray(X), dtype=p.dtype, device=p.device) - self.mu) / self.sd
        z = self.linear(Xs)
        if self.kind == "regression":
            z = z * self.y_sd + self.y_mu
        return z.cpu().numpy()

    def predict(self, X):
        """Class index (multiclass), logits (multilabel) or predictions (regression)."""
        s = self.scores(X)
        return s.argmax(1) if self.kind == "multiclass" else s


def _targets(Y, kind, n_classes, balanced, dev, dtype):
    """Target tensor, per-entry weights (already divided by each target's labelled-row count)."""
    if kind == "multiclass":
        y = np.asarray(Y).astype(np.int64)
        n_classes = int(n_classes or y.max() + 1)
        counts = np.bincount(y, minlength=n_classes).astype(float)
        if balanced:
            w = len(y) / ((counts > 0).sum() * counts[y])  # sklearn "balanced", over the classes present
        else:
            w = np.ones(len(y))
        return torch.as_tensor(y, device=dev), torch.as_tensor(w / len(y), dtype=dtype, device=dev), n_classes, None
    Y = np.asarray(Y, dtype=float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    known = np.isfinite(Y)
    n = np.maximum(known.sum(0), 1)
    Yz = np.where(known, Y, 0.0)
    y_stats = None
    if kind == "multilabel":
        Yz = (Yz > 0).astype(float)
        if balanced:
            n_pos = (Yz * known).sum(0)
            n_neg = known.sum(0) - n_pos
            w = np.where(Yz > 0, n / np.maximum(2 * n_pos, 1), n / np.maximum(2 * n_neg, 1))
        else:
            w = np.ones_like(Yz)
    else:
        mu = np.where(known, Y, 0).sum(0) / n
        sd = np.sqrt((np.where(known, Y - mu, 0) ** 2).sum(0) / n)
        sd[sd == 0] = 1
        Yz = np.where(known, (Y - mu) / sd, 0.0)
        w = np.ones_like(Yz)
        y_stats = (torch.as_tensor(mu, dtype=dtype, device=dev), torch.as_tensor(sd, dtype=dtype, device=dev))
    w = w * known / n
    return (torch.as_tensor(Yz, dtype=dtype, device=dev), torch.as_tensor(w, dtype=dtype, device=dev),
            Y.shape[1], y_stats)


def _train(linear, Xs, T, w, kind, l2, max_iter):
    opt = torch.optim.LBFGS(linear.parameters(), lr=1, max_iter=max_iter, history_size=20,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        z = linear(Xs)
        if kind == "multiclass":
            loss = (w * F.cross_entropy(z, T, reduction="none")).sum()
        elif kind == "multilabel":
            loss = (w * F.binary_cross_entropy_with_logits(z, T, reduction="none")).sum()
        else:
            loss = 0.5 * (w * (z - T) ** 2).sum()
        loss = loss + 0.5 * l2 * (linear.weight ** 2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    return linear


def fit(X, Y, kind, l2, n_classes=None, balanced=True, device="auto", dtype=torch.float32, max_iter=MAX_ITER,
        init=None):
    """Fit one probe with a fixed l2.

    X: (n, d) features. Y: multiclass = (n,) integer class indices 0..K-1; multilabel = (n, k) 0/1 with NaN
    for unknown; regression = (n,) or (n, k) floats with NaN for unknown. `init`: a LinearProbe to warm-start from.
    """
    assert kind in KINDS, kind
    dev = _device(device)
    X = torch.as_tensor(np.asarray(X), dtype=dtype, device=dev)
    mu, sd = X.mean(0), X.std(0, unbiased=False)
    sd = torch.where(sd > 0, sd, torch.ones_like(sd))
    Xs = (X - mu) / sd
    T, w, k, y_stats = _targets(Y, kind, n_classes, balanced, dev, dtype)
    linear = torch.nn.Linear(X.shape[1], k).to(device=dev, dtype=dtype)
    with torch.no_grad():
        if init is not None:
            linear.load_state_dict(init.linear.state_dict())
        else:
            linear.weight.zero_()
            linear.bias.zero_()
    _train(linear, Xs, T, w, kind, l2, max_iter)
    linear.requires_grad_(False)
    probe = LinearProbe(kind, linear, mu, sd, *(y_stats or (None, None)))
    probe.l2 = l2
    return probe


# ---- validation metrics (higher is better) -------------------------------------------------------------------

def balanced_accuracy(y, scores):
    from sklearn.metrics import balanced_accuracy_score

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return balanced_accuracy_score(y, scores.argmax(1))


def mean_auroc(Y, scores):
    """Mean ROC AUC over the columns that have both classes among their labelled rows."""
    from sklearn.metrics import roc_auc_score

    Y = np.asarray(Y, dtype=float)
    Y, scores = (Y[:, None], scores) if Y.ndim == 1 else (Y, scores)
    aucs = []
    for j in range(Y.shape[1]):
        m = np.isfinite(Y[:, j])
        yj = Y[m, j] > 0
        if 0 < yj.sum() < len(yj):
            aucs.append(roc_auc_score(yj, scores[m, j]))
    return float(np.mean(aucs)) if aucs else np.nan


def mean_r2(Y, pred):
    Y = np.asarray(Y, dtype=float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    r2 = []
    for j in range(Y.shape[1]):
        m = np.isfinite(Y[:, j])
        if m.sum() >= 2 and Y[m, j].std() > 0:
            r2.append(1 - ((Y[m, j] - pred[m, j]) ** 2).sum() / ((Y[m, j] - Y[m, j].mean()) ** 2).sum())
    return float(np.mean(r2)) if r2 else np.nan


METRICS = dict(multiclass=balanced_accuracy, multilabel=mean_auroc, regression=mean_r2)


def fit_select(X, Y, kind, n_classes=None, grid=L2_GRID, n_folds=3, seed=0, metric=None, **kw):
    """Choose l2 by K-fold CV on (X, Y), then refit on all of it. Returns the LinearProbe (.l2, .cv_scores)."""
    from sklearn.model_selection import KFold, StratifiedKFold

    metric = metric or METRICS[kind]
    X, Y = np.asarray(X), np.asarray(Y)
    if kind == "multiclass":
        n_classes = int(n_classes or Y.max() + 1)
        with warnings.catch_warnings():  # classes with fewer rows than folds
            warnings.simplefilter("ignore")
            folds = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, Y))
    else:
        folds = list(KFold(n_folds, shuffle=True, random_state=seed).split(X))
    cv = np.full((len(grid), len(folds)), np.nan)
    for f, (a, b) in enumerate(folds):
        prev = None
        for g, l2 in enumerate(grid):
            prev = fit(X[a], Y[a], kind, l2, n_classes=n_classes, init=prev, **kw)
            cv[g, f] = metric(Y[b], prev.scores(X[b]))
    mean = np.nanmean(cv, 1)
    best = int(np.nanargmax(mean)) if np.isfinite(mean).any() else 0
    probe = fit(X, Y, kind, grid[best], n_classes=n_classes, **kw)
    probe.cv_scores = dict(zip(grid, mean.tolist()))
    return probe
