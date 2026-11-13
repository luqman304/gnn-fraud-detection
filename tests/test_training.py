import numpy as np
import pandas as pd
import pytest

from gnn_fraud.experiment import BenchmarkConfig, run_benchmark
from gnn_fraud.metrics import illicit_metrics
from gnn_fraud.training import MODEL_KINDS, TrainedModel, train_model


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_every_model_learns_the_synthetic_signal(kind, graph, split, gnn_cfg, rf_cfg):
    model = train_model(kind, graph, split, "all", seed=0, gnn_cfg=gnn_cfg, rf_cfg=rf_cfg)
    prob = model.predict_proba(graph)
    assert prob.shape == (graph.num_nodes,)
    assert np.all((prob >= 0) & (prob <= 1))

    test = split.masks(graph)["test"]
    base_rate = graph.y[test].mean()
    assert illicit_metrics(graph.y[test], prob[test])["pr_auc"] > base_rate + 0.15


def test_gnn_training_is_reproducible(graph, split, gnn_cfg):
    a = train_model("sage", graph, split, "local", seed=3, gnn_cfg=gnn_cfg).predict_proba(graph)
    b = train_model("sage", graph, split, "local", seed=3, gnn_cfg=gnn_cfg).predict_proba(graph)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_gnn_records_selected_epoch_and_refits(graph, split, gnn_cfg):
    model = train_model("gcn", graph, split, "all", seed=0, gnn_cfg=gnn_cfg)
    assert 1 <= model.info["best_epoch"] <= gnn_cfg.max_epochs
    assert model.info["trained_on_steps"] == [1, split.val_end]
    # the scaler of the refit model must not have seen test-period features
    features = graph.features("all")
    expected = features[graph.time_step <= split.val_end].mean(axis=0)
    np.testing.assert_allclose(model.scaler.mean, expected, rtol=1e-5, atol=1e-6)


def test_labels_of_test_nodes_do_not_influence_training(graph, split, gnn_cfg, rf_cfg):
    test = split.masks(graph)["test"]
    flipped = type(graph)(**{**graph.__dict__, "y": graph.y.copy()})
    flipped.y[test] = 1 - flipped.y[test]
    for kind in ("rf", "sage"):
        a = train_model(kind, graph, split, "all", 0, gnn_cfg, rf_cfg).predict_proba(graph)
        b = train_model(kind, flipped, split, "all", 0, gnn_cfg, rf_cfg).predict_proba(flipped)
        np.testing.assert_allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("kind", ["rf", "sage", "rf+sage"])
def test_save_load_round_trip(kind, graph, split, gnn_cfg, rf_cfg, tmp_path):
    model = train_model(kind, graph, split, "local", seed=1, gnn_cfg=gnn_cfg, rf_cfg=rf_cfg)
    model.save(tmp_path / kind)
    loaded = TrainedModel.load(tmp_path / kind)
    np.testing.assert_allclose(loaded.predict_proba(graph), model.predict_proba(graph), atol=1e-6)
    assert loaded.info == model.info


def test_unknown_model_kind(graph, split):
    with pytest.raises(ValueError):
        train_model("xgboost", graph, split)


def test_benchmark_writes_results(graph, split, gnn_cfg, rf_cfg, tmp_path):
    cfg = BenchmarkConfig(
        models=("lr", "gcn"),
        feature_sets=("local",),
        seeds=(0, 1),
        split=split,
        gnn=gnn_cfg,
        rf=rf_cfg,
        shutdown_step=11,
        n_bootstrap=50,
    )
    summary = run_benchmark(graph, cfg, tmp_path, log=lambda _: None)
    assert list(summary["model"]) == ["lr", "gcn"]
    assert list(summary["n_runs"]) == [1, 2]
    for name in ("runs.csv", "per_time_step.csv", "summary.csv", "summary.md", "config.json"):
        assert (tmp_path / name).exists()
    assert "| local | gcn |" in (tmp_path / "summary.md").read_text()


def test_tuning_uses_validation_only_and_feeds_benchmark(graph, split, gnn_cfg, tmp_path):
    from gnn_fraud.experiment import load_tuned, tune_gnns

    grid = {"lr": (0.01, 0.003), "illicit_weight": (0.7,)}
    chosen = tune_gnns(graph, split, tmp_path, kinds=("mlp",), base=gnn_cfg, grid=grid, log=lambda _: None)
    assert set(chosen) == {"mlp"} and chosen["mlp"]["lr"] in grid["lr"]

    trials = pd.read_csv(tmp_path / "tuning.csv")
    assert len(trials) == 2 and trials["val_pr_auc"].between(0, 1).all()

    tuned = load_tuned(tmp_path / "gnn_hparams.json", gnn_cfg)
    cfg = BenchmarkConfig(gnn=gnn_cfg, gnn_by_kind=tuned)
    assert cfg.gnn_for("mlp").lr == chosen["mlp"]["lr"]
    assert cfg.gnn_for("gcn") is gnn_cfg
    assert load_tuned(tmp_path / "missing.json", gnn_cfg) == {}


def test_benchmark_resumes_from_checkpoints(graph, split, gnn_cfg, rf_cfg, tmp_path):
    cfg = BenchmarkConfig(
        models=("sage", "rf+sage"),
        feature_sets=("all",),
        seeds=(0,),
        split=split,
        gnn=gnn_cfg,
        rf=rf_cfg,
        shutdown_step=11,
        n_bootstrap=20,
    )
    first = run_benchmark(graph, cfg, tmp_path, log=lambda _: None)
    assert (tmp_path / "runs" / "sage-all-s0").is_dir()
    lines: list[str] = []
    second = run_benchmark(graph, cfg, tmp_path, log=lines.append)
    assert all(line.endswith("(checkpoint)") for line in lines) and len(lines) == 2
    pd.testing.assert_frame_equal(first, second)
