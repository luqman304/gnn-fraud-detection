import numpy as np
import pytest

from gnn_fraud.data import ILLICIT, LICIT, UNKNOWN, TxGraph, build_graph
from gnn_fraud.synthetic import write_elliptic_csvs


def test_build_graph_round_trips_elliptic_csv_layout(graph, tmp_path):
    write_elliptic_csvs(graph, tmp_path / "raw")
    parsed = build_graph(tmp_path / "raw", n_local=graph.n_local)

    np.testing.assert_array_equal(parsed.tx_id, graph.tx_id)
    np.testing.assert_array_equal(parsed.y, graph.y)
    np.testing.assert_array_equal(parsed.time_step, graph.time_step)
    np.testing.assert_array_equal(parsed.edge_index, graph.edge_index)
    np.testing.assert_allclose(parsed.x, graph.x, rtol=1e-6)


def test_build_graph_rejects_edges_to_unknown_transactions(graph, tmp_path):
    raw = tmp_path / "raw"
    write_elliptic_csvs(graph, raw)
    with open(raw / "elliptic_txs_edgelist.csv", "a") as f:
        f.write(f"{graph.tx_id[0]},123456789012\n")
    with pytest.raises(ValueError, match="unknown transaction"):
        build_graph(raw, n_local=graph.n_local)


def test_save_and_load(graph, tmp_path):
    graph.save(tmp_path / "g.npz")
    loaded = TxGraph.load(tmp_path / "g.npz")
    assert loaded.summary() == graph.summary()
    np.testing.assert_array_equal(loaded.x, graph.x)


def test_feature_sets(graph):
    assert graph.features("local").shape[1] == graph.n_local
    assert graph.features("all").shape[1] == graph.x.shape[1]
    with pytest.raises(ValueError):
        graph.features("everything")


def test_temporal_split_is_ordered_disjoint_and_labelled_only(graph, split):
    masks = split.masks(graph)
    ts = graph.time_step
    assert ts[masks["train"]].max() < ts[masks["val"]].min()
    assert ts[masks["val"]].max() < ts[masks["test"]].min()
    assert not (masks["train"] & masks["test"]).any()
    assert not (masks["val"] & masks["test"]).any()
    np.testing.assert_array_equal(masks["fit"], masks["train"] | masks["val"])
    for m in masks.values():
        assert set(np.unique(graph.y[m])) <= {ILLICIT, LICIT}
    assert (graph.y == UNKNOWN).sum() > 0


def test_synthetic_graph_has_no_cross_time_step_edges(graph):
    assert graph.summary()["cross_time_step_edges"] == 0
