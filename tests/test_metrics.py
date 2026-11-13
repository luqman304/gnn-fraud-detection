import numpy as np
import pytest

from gnn_fraud.metrics import illicit_metrics, mean_ci, per_time_step, time_block_bootstrap


def test_illicit_metrics_known_values():
    y = np.array([1, 1, 0, 0, 0])
    prob = np.array([0.9, 0.2, 0.8, 0.1, 0.0])
    m = illicit_metrics(y, prob)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert 0 < m["pr_auc"] <= 1


def test_mean_ci():
    mean, half = mean_ci([0.5, 0.5, 0.5])
    assert (mean, half) == (0.5, 0.0)
    mean, half = mean_ci([0.4, 0.5, 0.6])
    assert mean == pytest.approx(0.5)
    # t(0.975, df=2) * s / sqrt(n) = 4.303 * 0.1 / 1.732
    assert half == pytest.approx(0.2484, abs=1e-3)


def test_per_time_step_counts():
    y = np.array([1, 0, 1, 0])
    prob = np.array([0.9, 0.1, 0.1, 0.1])
    ts = np.array([1, 1, 2, 2])
    out = per_time_step(y, prob, ts)
    assert out[1]["f1"] == 1.0
    assert out[2]["f1"] == 0.0
    assert out[2]["n_illicit"] == 1


def test_time_block_bootstrap_brackets_point_estimate():
    rng = np.random.default_rng(0)
    ts = np.repeat(np.arange(10), 50)
    y = (rng.random(500) < 0.2).astype(int)
    prob = np.clip(y * 0.6 + rng.random(500) * 0.5, 0, 1)
    lo, hi = time_block_bootstrap(y, prob, ts, n_boot=200)
    point = illicit_metrics(y, prob)["f1"]
    assert lo <= point <= hi


def test_paired_block_bootstrap_detects_a_better_model():
    from gnn_fraud.metrics import paired_block_bootstrap

    rng = np.random.default_rng(1)
    ts = np.repeat(np.arange(12), 80)
    y = (rng.random(ts.size) < 0.2).astype(int)
    good = [np.clip(y * 0.7 + rng.random(ts.size) * 0.4, 0, 1) for _ in range(3)]
    bad = [rng.random(ts.size) for _ in range(3)]
    out = paired_block_bootstrap(y, good, bad, ts, n_boot=200)
    assert out["f1_diff"] > 0.3 and out["ci_lo"] > 0 and out["p_le_zero"] == 0.0
    same = paired_block_bootstrap(y, good, good, ts, n_boot=50)
    assert same["f1_diff"] == 0.0 and same["p_le_zero"] == 1.0
