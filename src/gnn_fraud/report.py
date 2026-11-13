"""Figure for the README: illicit F1 per test time step, with the dark-market shutdown marked."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import ILLICIT, TemporalSplit, TxGraph

SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SURFACE, TEXT, MUTED, GRID, BARS = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
DEFAULT_MODELS = ("rf", "mlp", "sage", "gcn")
LABELS = {
    "lr": "Logistic regression",
    "rf": "Random forest",
    "mlp": "MLP",
    "gcn": "GCN",
    "sage": "GraphSAGE",
    "rf+sage": "RF + GraphSAGE embeddings",
}


def plot_temporal(
    results_dir: Path,
    graph: TxGraph,
    split: TemporalSplit,
    features: str = "all",
    models: tuple[str, ...] = DEFAULT_MODELS,
    shutdown_step: int = 43,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = pd.read_csv(results_dir / "per_time_step.csv")
    steps = steps[steps["features"] == features]
    models = tuple(m for m in models if m in set(steps["model"]))[: len(SERIES_COLOURS)]

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED})
    fig, (ax, ax_n) = plt.subplots(
        2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}, facecolor=SURFACE
    )
    for a in (ax, ax_n):
        a.set_facecolor(SURFACE)
        a.tick_params(colors=MUTED, length=0)
        a.grid(axis="y", color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right", "left"):
            a.spines[side].set_visible(False)
        a.axvline(shutdown_step - 0.5, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))

    for model, colour in zip(models, SERIES_COLOURS, strict=False):
        mean = steps[steps["model"] == model].groupby("time_step")["f1"].mean()
        ax.plot(
            mean.index, mean.values, color=colour, linewidth=2, marker="o", markersize=4, label=LABELS[model]
        )
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Illicit F1 (mean over seeds)")
    ax.text(shutdown_step - 0.3, 1.0, "dark market shutdown", color=MUTED, fontsize=8, va="top")
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=TEXT)
    ax.set_title(
        f"Test time steps {split.val_end + 1}-{int(graph.time_step.max())}, {features} features",
        loc="left",
        color=TEXT,
        fontsize=11,
    )

    test_steps = np.arange(split.val_end + 1, int(graph.time_step.max()) + 1)
    counts = [int(((graph.time_step == t) & (graph.y == ILLICIT)).sum()) for t in test_steps]
    ax_n.bar(test_steps, counts, color=BARS, width=0.8)
    ax_n.set_ylabel("Illicit\ntransactions")
    ax_n.set_xlabel("Time step")
    ax_n.set_xticks(test_steps)

    fig.tight_layout()
    out = results_dir / f"temporal_f1_{features}.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out
