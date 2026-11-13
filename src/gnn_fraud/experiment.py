"""Benchmark: every model x feature set x seed, evaluated on the temporal test split."""

from __future__ import annotations

import itertools
import json
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import torch_geometric

from .data import FEATURE_SETS, TemporalSplit, TxGraph
from .metrics import illicit_metrics, mean_ci, paired_block_bootstrap, per_time_step, time_block_bootstrap
from .models import GNN_KINDS
from .training import MODEL_KINDS, GNNConfig, RFConfig, TrainedModel, train_gnn, train_model

DETERMINISTIC_KINDS = {"lr"}
METRICS = ("precision", "recall", "f1", "pr_auc", "roc_auc")
DARK_MARKET_SHUTDOWN_STEP = 43

# (model, features) pairs compared with a paired time-step bootstrap; A is listed first.
COMPARISONS = (
    (("rf", "all"), ("rf", "local")),
    (("rf", "all"), ("sage", "all")),
    (("rf", "all"), ("rf+sage", "all")),
    (("rf", "local"), ("rf+sage", "local")),
    (("mlp", "local"), ("sage", "local")),
    (("sage", "local"), ("gcn", "local")),
    (("sage", "all"), ("mlp", "all")),
    (("mlp", "local"), ("mlp", "all")),
)


@dataclass
class BenchmarkConfig:
    models: Sequence[str] = MODEL_KINDS
    feature_sets: Sequence[str] = FEATURE_SETS
    seeds: Sequence[int] = (0, 1, 2, 3, 4)
    split: TemporalSplit = field(default_factory=TemporalSplit)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    gnn_by_kind: dict[str, GNNConfig] = field(default_factory=dict)
    rf: RFConfig = field(default_factory=RFConfig)
    shutdown_step: int = DARK_MARKET_SHUTDOWN_STEP
    n_bootstrap: int = 1000

    def gnn_for(self, kind: str) -> GNNConfig:
        """Hyperparameters for a neural model; ``rf+sage`` uses GraphSAGE's."""
        return self.gnn_by_kind.get("sage" if kind == "rf+sage" else kind, self.gnn)


TUNING_GRID = {"lr": (0.01, 0.003), "illicit_weight": (0.5, 0.7)}


def tune_gnns(
    graph: TxGraph,
    split: TemporalSplit,
    out_dir: Path,
    kinds: Sequence[str] = GNN_KINDS,
    feature_set: str = "all",
    seed: int = 0,
    base: GNNConfig | None = None,
    grid: dict[str, Sequence[Any]] = TUNING_GRID,
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, Any]]:
    """Grid search on the validation steps only (train ``1..train_end``, score ``..val_end``).

    Writes every trial to ``tuning.csv`` and the best setting per model kind to
    ``gnn_hparams.json``, which ``eval`` and ``train`` pick up automatically.
    """
    base = base or GNNConfig()
    keys = list(grid)
    trials, best = [], {}
    for kind in kinds:
        for values in itertools.product(*(grid[k] for k in keys)):
            params = dict(zip(keys, values, strict=True))
            cfg = replace(base, **params)
            model = train_gnn(kind, graph, split, feature_set, seed, cfg, refit=False)
            trial = {"model": kind, **params, **{k: model.info[k] for k in ("val_pr_auc", "best_epoch")}}
            trials.append(trial)
            log(f"{kind:5s} {params} val_pr_auc={trial['val_pr_auc']:.4f} best_epoch={trial['best_epoch']}")
            if kind not in best or trial["val_pr_auc"] > best[kind]["val_pr_auc"]:
                best[kind] = trial
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trials).to_csv(out_dir / "tuning.csv", index=False)
    chosen = {kind: {k: t[k] for k in keys} for kind, t in best.items()}
    (out_dir / "gnn_hparams.json").write_text(json.dumps(chosen, indent=2))
    return chosen


def load_tuned(path: Path, base: GNNConfig) -> dict[str, GNNConfig]:
    if not path.exists():
        return {}
    return {kind: replace(base, **params) for kind, params in json.loads(path.read_text()).items()}


def run_benchmark(
    graph: TxGraph,
    cfg: BenchmarkConfig,
    out_dir: Path,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Train and evaluate every configuration, write raw and summary results to ``out_dir``.

    Each finished run is checkpointed under ``out_dir/runs`` (test-set probabilities, plus
    the GraphSAGE weights reused by ``rf+sage``), so an interrupted benchmark resumes
    where it stopped. Returns the summary table (one row per model and feature set).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "runs"
    ckpt_dir.mkdir(exist_ok=True)
    test = cfg.split.masks(graph)["test"]
    y_test = graph.y[test].astype(int)
    ts_test = graph.time_step[test]
    before = ts_test < cfg.shutdown_step

    runs: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    all_probs: dict[tuple[str, str], list[np.ndarray]] = {}
    for feature_set in cfg.feature_sets:
        for kind in cfg.models:
            seeds = list(cfg.seeds)[:1] if kind in DETERMINISTIC_KINDS else list(cfg.seeds)
            first_prob = None
            for seed in seeds:
                ckpt = ckpt_dir / f"{kind}-{feature_set}-s{seed}.npz"
                if ckpt.exists():
                    with np.load(ckpt) as f:
                        prob, threshold = f["prob"], float(f["threshold"])
                        seconds, best_epoch = float(f["seconds"]), int(f["best_epoch"])
                    resumed = " (checkpoint)"
                else:
                    start = time.perf_counter()
                    sage_dir = ckpt_dir / f"sage-{feature_set}-s{seed}"
                    cached = TrainedModel.load(sage_dir) if kind == "rf+sage" and sage_dir.exists() else None
                    model = train_model(
                        kind, graph, cfg.split, feature_set, seed, cfg.gnn_for(kind), cfg.rf, cached
                    )
                    if kind == "sage":
                        model.save(sage_dir)
                    prob = model.predict_proba(graph)[test]
                    seconds = time.perf_counter() - start
                    threshold = model.threshold
                    info = model.embedder.info if model.embedder is not None else model.info
                    best_epoch = int(info.get("best_epoch", -1))
                    np.savez(ckpt, prob=prob, threshold=threshold, seconds=seconds, best_epoch=best_epoch)
                    resumed = ""
                first_prob = prob if first_prob is None else first_prob
                all_probs.setdefault((kind, feature_set), []).append(prob)

                row = {"model": kind, "features": feature_set, "seed": seed, "seconds": round(seconds, 2)}
                row.update(illicit_metrics(y_test, prob, threshold))
                for name, part in (("before", before), ("after", ~before)):
                    row[f"f1_{name}_shutdown"] = (
                        illicit_metrics(y_test[part], prob[part], threshold)["f1"]
                        if part.any()
                        else float("nan")
                    )
                row["best_epoch"] = best_epoch if best_epoch > 0 else None
                runs.append(row)
                for t, m in per_time_step(y_test, prob, ts_test, threshold).items():
                    steps.append({"model": kind, "features": feature_set, "seed": seed, "time_step": t, **m})
                log(
                    f"{kind:8s} {feature_set:5s} seed={seed} f1={row['f1']:.3f} "
                    f"pr_auc={row['pr_auc']:.3f} ({seconds:.0f}s){resumed}"
                )

            seed_runs = [r for r in runs if r["model"] == kind and r["features"] == feature_set]
            agg: dict[str, Any] = {"model": kind, "features": feature_set, "n_runs": len(seed_runs)}
            for metric in (*METRICS, "f1_before_shutdown", "f1_after_shutdown"):
                agg[metric], agg[f"{metric}_ci"] = mean_ci([r[metric] for r in seed_runs])
            assert first_prob is not None
            agg["f1_bootstrap_lo"], agg["f1_bootstrap_hi"] = time_block_bootstrap(
                y_test, first_prob, ts_test, "f1", n_boot=cfg.n_bootstrap
            )
            summary.append(agg)

    comparisons = []
    for a, b in COMPARISONS:
        if a in all_probs and b in all_probs:
            result = paired_block_bootstrap(
                y_test, all_probs[a], all_probs[b], ts_test, n_boot=cfg.n_bootstrap
            )
            comparisons.append({"a": "/".join(a), "b": "/".join(b), **result})
    if comparisons:
        comp_df = pd.DataFrame(comparisons)
        comp_df.to_csv(out_dir / "comparisons.csv", index=False)
        (out_dir / "comparisons.md").write_text(comparisons_markdown(comp_df))

    runs_df, steps_df, summary_df = pd.DataFrame(runs), pd.DataFrame(steps), pd.DataFrame(summary)
    runs_df.to_csv(out_dir / "runs.csv", index=False)
    steps_df.to_csv(out_dir / "per_time_step.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.md").write_text(summary_markdown(summary_df))
    (out_dir / "config.json").write_text(json.dumps(_provenance(graph, cfg), indent=2, default=str))
    return summary_df


def _fmt(mean: float, ci: float) -> str:
    return f"{mean:.3f} ± {ci:.3f}" if ci > 0 else f"{mean:.3f}"


def summary_markdown(summary: pd.DataFrame) -> str:
    """Render the summary as a Markdown table (mean ± 95% CI half-width over seeds)."""
    header = (
        "| Features | Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | F1 95% CI (time-step bootstrap) "
        "| F1 before step 43 | F1 from step 43 |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['features']} | {r['model']} | {_fmt(r['precision'], r['precision_ci'])} "
            f"| {_fmt(r['recall'], r['recall_ci'])} | **{_fmt(r['f1'], r['f1_ci'])}** "
            f"| {_fmt(r['pr_auc'], r['pr_auc_ci'])} | {_fmt(r['roc_auc'], r['roc_auc_ci'])} "
            f"| [{r['f1_bootstrap_lo']:.3f}, {r['f1_bootstrap_hi']:.3f}] "
            f"| {_fmt(r['f1_before_shutdown'], r['f1_before_shutdown_ci'])} "
            f"| {_fmt(r['f1_after_shutdown'], r['f1_after_shutdown_ci'])} |"
        )
    return header + "\n".join(lines) + "\n"


def comparisons_markdown(comparisons: pd.DataFrame) -> str:
    lines = [
        "| A (model/features) | B | F1 difference A - B | 95% CI (paired time-step bootstrap) | Share of resamples with A ≤ B |",
        "|---|---|---|---|---|",
    ]
    for _, r in comparisons.iterrows():
        lines.append(
            f"| {r['a']} | {r['b']} | {r['f1_diff']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] "
            f"| {r['p_le_zero']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _provenance(graph: TxGraph, cfg: BenchmarkConfig) -> dict[str, Any]:
    return {
        "config": asdict(cfg),
        "graph": graph.summary(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "platform": platform.platform(),
    }
