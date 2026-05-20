"""figures: confusion heatmaps, model comparison, label mix, subset accuracy."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from labels import LABEL_ORDER, NEG, NEU, POS

SUBSET_ORDER: tuple[str, ...] = (
    "common_100",
    "annotator_1_unique",
    "annotator_2_unique",
    "annotator_3_unique",
)


def subset_accuracy_vector(records: list[dict[str, Any]]) -> np.ndarray:
    v: list[float] = []
    for sub in SUBSET_ORDER:
        rs = [r for r in records if r["subset"] == sub]
        if not rs:
            v.append(float("nan"))
        else:
            v.append(sum(1 for r in rs if r["correct"]) / len(rs))
    return np.array(v, dtype=float)
_CMAP_COUNTS = "Blues"
_CMAP_NORM = "YlOrBr"


def _annotate_heatmap(
    ax: Any,
    data: np.ndarray,
    fmt: str,
    text_color_thresh: float | None = None,
) -> None:
    h, w = data.shape
    if text_color_thresh is None:
        text_color_thresh = (float(np.min(data)) + float(np.max(data))) / 2 if data.size else 0.0
    for i in range(h):
        for j in range(w):
            val = data[i, j]
            if fmt == "d":
                s = str(int(round(float(val))))
            else:
                s = format(float(val), fmt)
            color = "white" if float(val) > text_color_thresh else "black"
            ax.text(j, i, s, ha="center", va="center", color=color, fontsize=11)


def confusion_heatmap_counts(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    title: str,
    out_path: Path,
) -> None:
    cm = confusion_matrix(list(y_true), list(y_pred), labels=list(LABEL_ORDER))
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap=_CMAP_COUNTS, vmin=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([NEG, NEU, POS], rotation=25, ha="right")
    ax.set_yticklabels([NEG, NEU, POS])
    ax.set_ylabel("gold")
    ax.set_xlabel("predicted")
    ax.set_title(title)
    _annotate_heatmap(ax, cm.astype(float), "d", text_color_thresh=cm.max() / 2 if cm.size else 0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def confusion_heatmap_row_norm(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    title: str,
    out_path: Path,
) -> None:
    cm = confusion_matrix(list(y_true), list(y_pred), labels=list(LABEL_ORDER))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    rn = cm.astype(float) / row_sums
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(rn, cmap=_CMAP_NORM, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([NEG, NEU, POS], rotation=25, ha="right")
    ax.set_yticklabels([NEG, NEU, POS])
    ax.set_ylabel("gold")
    ax.set_xlabel("predicted")
    ax.set_title(title + " (row-normalized)")
    _annotate_heatmap(ax, rn, ".2f", text_color_thresh=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def grid_row_norm_confusions(
    panels: list[tuple[str, list[str], list[str]]],
    out_path: Path,
    suptitle: str = "model comparison — row-normalized confusion",
) -> None:
    n = len(panels)
    if n == 0:
        return
    from matplotlib.colors import Normalize

    if n <= 3:
        nrows, ncols = 1, n
    else:
        ncols = 3
        nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols + 1.2, 3.9 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    for j in range(len(axes_flat)):
        if j >= n:
            axes_flat[j].set_visible(False)

    norm = Normalize(vmin=0, vmax=1)
    last_im = None
    for idx, (name, yt, yp) in enumerate(panels):
        ax = axes_flat[idx]
        cm = confusion_matrix(list(yt), list(yp), labels=list(LABEL_ORDER))
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        rn = cm.astype(float) / row_sums
        last_im = ax.imshow(rn, cmap=_CMAP_NORM, norm=norm)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(["neg", "neu", "pos"], fontsize=9)
        ax.set_yticklabels(["neg", "neu", "pos"], fontsize=9)
        ax.set_xlabel("pred")
        ax.set_ylabel("gold")
        ax.set_title(name, fontsize=10)
        _annotate_heatmap(ax, rn, ".2f", text_color_thresh=0.35)

    fig.suptitle(suptitle, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(right=0.88, top=0.92)
    cax = fig.add_axes([0.91, 0.2, 0.018, 0.55])
    fig.colorbar(last_im, cax=cax)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def metrics_grouped_bars(
    names: list[str],
    accuracies: list[float],
    macro_f1s: list[float],
    out_path: Path,
) -> None:
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(names)), 4))
    ax.bar(x - w / 2, accuracies, width=w, label="accuracy", color="#4477aa")
    ax.bar(x + w / 2, macro_f1s, width=w, label="macro F1", color="#cc6677")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.axhline(1 / 3, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(
        len(names) - 0.5,
        1 / 3 + 0.02,
        "random ~0.33",
        fontsize=8,
        color="gray",
    )
    ax.legend()
    ax.set_title("overall scores (1k gold, invalid preds = wrong for accuracy)")
    ax.set_ylabel("score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def label_mix_stacked_bars(
    model_name: str,
    gold_labels: Sequence[str],
    pred_display: Sequence[str],
    out_path: Path,
) -> None:
    """horizontal stacked bars: gold vs model class fractions (model includes invalid)."""
    colors = {
        NEG: "#c0392b",
        NEU: "#f39c12",
        POS: "#27ae60",
        "invalid": "#7f8c8d",
    }
    n_g = len(gold_labels) or 1
    n_p = len(pred_display) or 1
    gc = Counter(gold_labels)
    pc = Counter(pred_display)

    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    y_gold, y_model = 1.0, 0.35
    bar_h = 0.28
    legend_keys: list[str] = []

    def stack_bar(y: float, counts: Counter, keys: list[str], ntot: int) -> None:
        left = 0.0
        for k in keys:
            w = counts.get(k, 0) / ntot
            if w <= 0:
                continue
            show_leg = k not in legend_keys
            if show_leg:
                legend_keys.append(k)
            ax.barh(
                y,
                w,
                left=left,
                height=bar_h,
                color=colors.get(k, "#333"),
                label=k if show_leg else None,
                edgecolor="white",
                linewidth=0.5,
            )
            if w >= 0.07:
                ax.text(
                    left + w / 2,
                    y,
                    f"{w:.0%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if w > 0.12 else "black",
                )
            left += w

    stack_bar(y_gold, gc, list(LABEL_ORDER), n_g)
    stack_bar(y_model, pc, list(LABEL_ORDER) + ["invalid"], n_p)
    ax.set_yticks([y_gold, y_model])
    ax.set_yticklabels(["gold", "model"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of 1k items")
    ax.set_title(f"class mix — {model_name}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def subset_accuracy_heatmap(
    model_names: list[str],
    subset_accs: np.ndarray,
    out_path: Path,
) -> None:
    """
    subset_accs shape (n_models, n_subsets), values 0..1.
    row labels model_names, cols SUBSET_ORDER.
    """
    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.55 * len(model_names))))
    im = ax.imshow(subset_accs, cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="accuracy")
    ax.set_xticks(range(len(SUBSET_ORDER)))
    ax.set_xticklabels(["common\n100", "ann1\n300", "ann2\n300", "ann3\n300"], fontsize=9)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel("gold subset")
    ax.set_ylabel("model")
    ax.set_title("accuracy by subset (invalid = wrong)")
    for i in range(subset_accs.shape[0]):
        for j in range(subset_accs.shape[1]):
            ax.text(j, i, f"{subset_accs[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def per_class_f1_grouped(
    names: list[str],
    per_f1_rows: list[tuple[float, float, float]],
    out_path: Path,
) -> None:
    x = np.arange(len(names))
    w = 0.24
    fig, ax = plt.subplots(figsize=(max(5.5, 1.3 * len(names)), 4.2))
    for i, (lab, color) in enumerate(zip(LABEL_ORDER, ["#c0392b", "#2980b9", "#27ae60"])):
        vals = [row[i] for row in per_f1_rows]
        ax.bar(x + (i - 1) * w, vals, width=w, label=lab, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(title="class F1")
    ax.set_ylabel("F1 (valid preds only)")
    ax.set_title("per-class F1 by model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
