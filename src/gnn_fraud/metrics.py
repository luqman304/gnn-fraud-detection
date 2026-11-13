"""Illicit-class metrics, confidence intervals and per-time-step breakdowns."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def illicit_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Precision, recall and F1 for the illicit class at ``threshold``, plus ranking metrics."""
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(prob) >= threshold).astype(int)
    both_classes = 0 < y_true.sum() < len(y_true)
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, prob)) if both_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, prob)) if both_classes else float("nan"),
        "micro_f1": float(f1_score(y_true, pred, average="micro")),
    }


def mean_ci(values: list[float] | np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of a Student-t confidence interval over independent runs."""
    v = np.asarray(values, dtype=float)
    if v.size < 2 or np.allclose(v, v[0]):
        return float(v.mean()), 0.0
    half = stats.t.ppf((1 + confidence) / 2, df=v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
    return float(v.mean()), float(half)


def per_time_step(
    y_true: np.ndarray, prob: np.ndarray, time_step: np.ndarray, threshold: float = 0.5
) -> dict[int, dict[str, float]]:
    """Illicit F1 / precision / recall for each time step (labelled nodes only)."""
    out = {}
    for t in np.unique(time_step):
        m = time_step == t
        pred = prob[m] >= threshold
        out[int(t)] = {
            "f1": float(f1_score(y_true[m], pred, zero_division=0)),
            "precision": float(precision_score(y_true[m], pred, zero_division=0)),
            "recall": float(recall_score(y_true[m], pred, zero_division=0)),
            "n_illicit": int(y_true[m].sum()),
            "n_labelled": int(m.sum()),
        }
    return out


def time_block_bootstrap(
    y_true: np.ndarray,
    prob: np.ndarray,
    time_step: np.ndarray,
    metric: str = "f1",
    threshold: float = 0.5,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI that resamples whole test time steps.

    Nodes within a time step are strongly correlated (shared actors, shared market
    conditions), so resampling time steps rather than nodes gives an honest estimate
    of how much the score depends on which periods happen to be in the test set.
    """
    rng = np.random.default_rng(seed)
    steps = np.unique(time_step)
    groups = [np.flatnonzero(time_step == t) for t in steps]
    scores = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(steps), len(steps))])
        if y_true[idx].sum() == 0:
            continue
        scores.append(illicit_metrics(y_true[idx], prob[idx], threshold)[metric])
    lo, hi = np.quantile(scores, [(1 - confidence) / 2, (1 + confidence) / 2])
    return float(lo), float(hi)


def _f1(y_true: np.ndarray, pred: np.ndarray) -> float:
    tp = int((y_true & pred).sum())
    denom = 2 * tp + int((~y_true & pred).sum()) + int((y_true & ~pred).sum())
    return 2 * tp / denom if denom else 0.0


def paired_block_bootstrap(
    y_true: np.ndarray,
    probs_a: list[np.ndarray],
    probs_b: list[np.ndarray],
    time_step: np.ndarray,
    threshold: float = 0.5,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Paired comparison of two models' seed-averaged illicit F1 (A minus B).

    Both models are scored on the same resampled set of test time steps in every
    replicate, so period-to-period variation largely cancels out. ``p_le_zero`` is the
    share of replicates in which A does not beat B.
    """
    y = np.asarray(y_true).astype(bool)
    preds_a = [np.asarray(p) >= threshold for p in probs_a]
    preds_b = [np.asarray(p) >= threshold for p in probs_b]

    def diff(idx: np.ndarray) -> float:
        a = np.mean([_f1(y[idx], p[idx]) for p in preds_a])
        b = np.mean([_f1(y[idx], p[idx]) for p in preds_b])
        return float(a - b)

    rng = np.random.default_rng(seed)
    steps = np.unique(time_step)
    groups = [np.flatnonzero(time_step == t) for t in steps]
    diffs = np.array(
        [
            diff(np.concatenate([groups[i] for i in rng.integers(0, len(steps), len(steps))]))
            for _ in range(n_boot)
        ]
    )
    lo, hi = np.quantile(diffs, [(1 - confidence) / 2, (1 + confidence) / 2])
    return {
        "f1_diff": diff(np.arange(len(y))),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_le_zero": float((diffs <= 0).mean()),
    }
