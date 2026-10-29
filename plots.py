"""Figures: confusion matrix, training curves and model-comparison bar charts."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import LABEL_NAMES

COLORS = {"CNN": "#8DA0CB", "BiLSTM": "#66C2A5", "BERT": "#A6A6A6", "CB": "#E6862E", "BB": "#4C72B0",
          "HyPhiX": "#C44E52"}


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#E3E3E3", lw=0.8)
    ax.set_axisbelow(True)


def plot_confusion(cm, path, title=None, normalise=False, class_names=None, dpi=400):
    """Annotated confusion matrix (rows = true class, columns = predicted class)."""
    cm = np.asarray(cm, dtype=float)
    names = class_names or [f"{i}\n{n}" for i, n in enumerate(LABEL_NAMES)]
    data = cm / cm.sum(1, keepdims=True).clip(min=1) if normalise else cm
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(data, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    thr = data.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{data[i, j]:.2f}" if normalise else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", color="white" if data[i, j] > thr else "black", fontsize=11)
    ax.set_xticks(range(len(names)), names, fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_history(history, path, dpi=300):
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key, lab in [(axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")]:
        ax.plot(ep, [h[f"train_{key}"] for h in history], "o-", label="training")
        ax.plot(ep, [h[f"val_{key}"] for h in history], "s-", label="validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(lab)
        _style(ax)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_accuracy_bars(values: dict, path, ylabel="Accuracy", ylim=None, errors: dict | None = None, dpi=400):
    """Bar chart of one metric per model, optionally with error bars (e.g. SD over seeds)."""
    names = list(values)
    v = [values[n] for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    err = [errors.get(n, 0) for n in names] if errors else None
    bars = ax.bar(names, v, 0.55, color=[COLORS.get(n, "#9DB4D3") for n in names], yerr=err, capsize=4)
    for b, val in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.002, f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_grouped_metrics(table: dict, path, dpi=400):
    """table = {model: {"Precision": x, "Recall": y, "F1-score": z}} (macro averages)."""
    metrics = ["Precision", "Recall", "F1-score"]
    models = list(table)
    w = 0.8 / len(models)
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, m in enumerate(models):
        vals = [table[m][k] for k in metrics]
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * w, vals, w * 0.92, color=COLORS.get(m, None), label=m)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.002, f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7, rotation=90)
    ax.set_xticks(x, metrics)
    ax.set_ylabel("Score (macro average)")
    lo = min(min(t.values()) for t in table.values())
    ax.set_ylim(max(0.0, lo - 0.05), 1.0)
    ax.legend(ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
