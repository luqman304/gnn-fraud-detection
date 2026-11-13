import pytest

from gnn_fraud.experiment import BenchmarkConfig, run_benchmark

pytest.importorskip("matplotlib")


def test_temporal_figure_is_written(graph, split, gnn_cfg, rf_cfg, tmp_path):
    from gnn_fraud.report import plot_temporal

    cfg = BenchmarkConfig(
        models=("rf", "mlp"),
        feature_sets=("all",),
        seeds=(0,),
        split=split,
        gnn=gnn_cfg,
        rf=rf_cfg,
        n_bootstrap=20,
    )
    run_benchmark(graph, cfg, tmp_path, log=lambda _: None)
    out = plot_temporal(tmp_path, graph, split, shutdown_step=11)
    assert out.exists() and out.stat().st_size > 10_000
