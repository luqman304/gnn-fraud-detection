"""Command-line interface: ``gnnfraud prepare | train | eval | serve``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .config import get_paths
from .data import ELLIPTIC_BASE_URL, FEATURE_SETS, TemporalSplit, TxGraph, build_graph, download_raw

app = typer.Typer(help="Illicit transaction detection on the Elliptic Bitcoin graph.", no_args_is_help=True)


def _load_graph(graph_file: Path) -> TxGraph:
    if not graph_file.exists():
        typer.echo(f"{graph_file} not found; run `gnnfraud prepare` first.", err=True)
        raise typer.Exit(1)
    return TxGraph.load(graph_file)


def _check_choice(value: str, choices: tuple[str, ...], name: str) -> None:
    if value not in choices:
        raise typer.BadParameter(f"{value!r} is not one of {', '.join(choices)}", param_hint=name)


@app.command()
def prepare(
    raw_dir: Annotated[Path | None, typer.Option(help="Directory for the raw CSV files.")] = None,
    base_url: Annotated[str, typer.Option(help="Mirror hosting the zipped CSV files.")] = ELLIPTIC_BASE_URL,
    force: Annotated[bool, typer.Option(help="Re-download even if the CSV files exist.")] = False,
) -> None:
    """Download the Elliptic dataset and build the processed graph file."""
    paths = get_paths()
    raw = raw_dir or paths.raw_dir
    download_raw(raw, base_url=base_url, force=force)
    graph = build_graph(raw)
    graph.save(paths.graph_file)
    typer.echo(json.dumps(graph.summary(), indent=2))
    typer.echo(f"wrote {paths.graph_file}")


@app.command()
def train(
    model: Annotated[str, typer.Option(help="lr, rf, mlp, gcn, sage or rf+sage.")] = "sage",
    features: Annotated[str, typer.Option(help="local (93 features) or all (165).")] = "all",
    seed: int = 0,
    device: Annotated[str, typer.Option(help="cpu, cuda, mps or auto (neural models only).")] = "cpu",
    out: Annotated[Path | None, typer.Option(help="Artifact directory.")] = None,
) -> None:
    """Train one model on time steps 1-34, report test metrics and save it for serving."""
    from .experiment import BenchmarkConfig, load_tuned
    from .metrics import illicit_metrics
    from .training import MODEL_KINDS, GNNConfig, train_model

    _check_choice(model, MODEL_KINDS, "--model")
    _check_choice(features, FEATURE_SETS, "--features")
    paths = get_paths()
    graph = _load_graph(paths.graph_file)
    split = TemporalSplit()
    base = GNNConfig(device=device)
    cfg = BenchmarkConfig(gnn=base, gnn_by_kind=load_tuned(paths.results_dir / "gnn_hparams.json", base))
    trained = train_model(model, graph, split, features, seed, cfg.gnn_for(model))
    test = split.masks(graph)["test"]
    metrics = illicit_metrics(graph.y[test], trained.predict_proba(graph)[test], trained.threshold)
    trained.info["test_metrics"] = metrics
    target = out or paths.artifacts_dir / f"{model}-{features}"
    trained.save(target)
    typer.echo(json.dumps({"artifact": str(target), **trained.info}, indent=2))


@app.command()
def tune(
    device: Annotated[str, typer.Option(help="cpu, cuda, mps or auto.")] = "cpu",
    out: Annotated[Path | None, typer.Option(help="Results directory.")] = None,
) -> None:
    """Select GNN learning rate and class weight on the validation steps (30-34) only."""
    from .experiment import tune_gnns
    from .training import GNNConfig

    paths = get_paths()
    graph = _load_graph(paths.graph_file)
    chosen = tune_gnns(
        graph, TemporalSplit(), out or paths.results_dir, base=GNNConfig(device=device), log=typer.echo
    )
    typer.echo(json.dumps(chosen, indent=2))


@app.command("eval")
def evaluate(
    model: Annotated[list[str] | None, typer.Option(help="Models to run (repeatable). Default: all.")] = None,
    features: Annotated[
        list[str] | None, typer.Option(help="Feature sets (repeatable). Default: both.")
    ] = None,
    seeds: Annotated[int, typer.Option(help="Number of seeds per stochastic model.")] = 5,
    device: Annotated[str, typer.Option(help="cpu, cuda, mps or auto (neural models only).")] = "cpu",
    out: Annotated[Path | None, typer.Option(help="Results directory.")] = None,
    plot: Annotated[bool, typer.Option(help="Write the per-time-step figure (needs matplotlib).")] = True,
) -> None:
    """Run the full benchmark on the temporal test split and write results tables."""
    from .experiment import BenchmarkConfig, load_tuned, run_benchmark, summary_markdown
    from .training import MODEL_KINDS

    for m in model or []:
        _check_choice(m, MODEL_KINDS, "--model")
    for f in features or []:
        _check_choice(f, FEATURE_SETS, "--features")
    paths = get_paths()
    graph = _load_graph(paths.graph_file)
    results_dir = out or paths.results_dir
    cfg = BenchmarkConfig(seeds=tuple(range(seeds)))
    cfg.gnn.device = device
    cfg.gnn_by_kind = load_tuned(results_dir / "gnn_hparams.json", cfg.gnn)
    if model:
        cfg.models = tuple(model)
    if features:
        cfg.feature_sets = tuple(features)
    summary = run_benchmark(graph, cfg, results_dir, log=typer.echo)
    typer.echo(summary_markdown(summary))
    if plot:
        from .report import plot_temporal

        typer.echo(f"wrote {plot_temporal(results_dir, graph, cfg.split)}")


@app.command()
def serve(
    artifact: Annotated[Path | None, typer.Option(help="Artifact directory written by `train`.")] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Serve the scoring API for a trained model."""
    import uvicorn

    from .api import create_app

    paths = get_paths()
    uvicorn.run(create_app(paths.graph_file, artifact or paths.default_artifact), host=host, port=port)


if __name__ == "__main__":
    app()
