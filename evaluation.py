"""Metrics, confidence intervals, significance tests and case-level outputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from . import LABEL_NAMES


def wilson_interval(k: int, n: int, level: float = 0.95):
    """Wilson score interval for a binomial proportion k / n."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(0.5 + level / 2)
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (float(centre - half), float(centre + half))


def bootstrap_interval(y_true, y_pred, metric="macro_f1", n_resamples=10000, level=0.95, seed=2018):
    """Percentile bootstrap interval of a metric over the evaluation tweets."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    f = {"macro_f1": lambda a, b: f1_score(a, b, average="macro", labels=range(len(LABEL_NAMES)), zero_division=0),
         "accuracy": accuracy_score}[metric]
    vals = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        vals[i] = f(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(vals, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def compute_metrics(y_true, y_pred, n_bootstrap=10000, level=0.95, seed=2018) -> dict:
    """All metrics reported in the manuscript, computed from case-level predictions."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(LABEL_NAMES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n, k = len(y_true), int((y_true == y_pred).sum())
    per_class = []
    for c in labels:
        lo, hi = wilson_interval(int(cm[c, c]), int(cm[c].sum()), level)
        per_class.append({"class_index": c, "class": LABEL_NAMES[c], "support": int(s[c]),
                          "precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c]),
                          "recall_ci_low": lo, "recall_ci_high": hi})
    acc_lo, acc_hi = wilson_interval(k, n, level)
    out = {
        "n": n, "correct": k, "accuracy": k / n,
        "accuracy_ci_wilson": [acc_lo, acc_hi],
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": per_class, "confusion_matrix": cm.tolist(),
    }
    if n_bootstrap:
        out["macro_f1_ci_bootstrap"] = list(bootstrap_interval(y_true, y_pred, "macro_f1", n_bootstrap, level, seed))
    return out


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """Exact (binomial) McNemar test for two models evaluated on the same tweets."""
    y_true, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    ca, cb = a == y_true, b == y_true
    n01 = int((ca & ~cb).sum())   # A right, B wrong
    n10 = int((~ca & cb).sum())   # A wrong, B right
    m = n01 + n10
    p = float(stats.binomtest(min(n01, n10), m, 0.5).pvalue) if m else 1.0
    return {"a_correct_b_wrong": n01, "a_wrong_b_correct": n10, "discordant": m, "p_value": p}


def save_predictions(df: pd.DataFrame, probs: np.ndarray, path: Path, alphas=None) -> pd.DataFrame:
    """Case-level outputs: one row per evaluated tweet with label, prediction and probabilities."""
    out = pd.DataFrame({"tweet_id": df["tweet_id"].values, "entity": df["entity"].values,
                        "raw_text": df["raw_text"].values, "text": df["text"].values,
                        "true_label": df["label"].values, "true_class": df["sentiment"].values,
                        "pred_label": probs.argmax(1)})
    out["pred_class"] = [LABEL_NAMES[i] for i in out["pred_label"]]
    out["correct"] = out["true_label"] == out["pred_label"]
    for c, name in enumerate(LABEL_NAMES):
        out[f"prob_{name}"] = probs[:, c]
    if alphas is not None:
        for j in range(alphas.shape[1]):
            out[f"gate_alpha_{j + 1}"] = alphas[:, j]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def classification_table(metrics: dict) -> pd.DataFrame:
    rows = [{k: v for k, v in r.items()} for r in metrics["per_class"]]
    rows.append({"class_index": "", "class": "macro avg", "support": metrics["n"],
                 "precision": metrics["macro_precision"], "recall": metrics["macro_recall"],
                 "f1": metrics["macro_f1"], "recall_ci_low": "", "recall_ci_high": ""})
    return pd.DataFrame(rows)


def aggregate_seeds(results: list[dict], keys=("accuracy", "macro_f1", "macro_precision", "macro_recall"),
                    level=0.95) -> dict:
    """Mean, standard deviation and t-based interval of metrics over repeated runs."""
    out = {"n_runs": len(results)}
    for k in keys:
        v = np.array([r[k] for r in results], dtype=float)
        sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        half = float(stats.t.ppf(0.5 + level / 2, len(v) - 1) * sd / np.sqrt(len(v))) if len(v) > 1 else 0.0
        out[k] = {"mean": float(v.mean()), "sd": sd, "ci": [float(v.mean() - half), float(v.mean() + half)],
                  "values": v.tolist()}
    return out


def paired_seed_test(a: list[float], b: list[float]) -> dict:
    """Paired t-test and Wilcoxon signed-rank test of two models over the same seeds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = stats.ttest_rel(a, b)
    res = {"mean_difference": float((a - b).mean()), "t_statistic": float(t.statistic), "t_p_value": float(t.pvalue)}
    try:
        w = stats.wilcoxon(a, b)
        res.update({"wilcoxon_statistic": float(w.statistic), "wilcoxon_p_value": float(w.pvalue)})
    except ValueError:
        res.update({"wilcoxon_statistic": None, "wilcoxon_p_value": None})
    return res
