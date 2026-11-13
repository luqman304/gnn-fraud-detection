"""Small synthetic transaction graphs with the Elliptic layout, for tests and demos.

Illicit nodes have a weak shift in a few of their own features and preferentially
transact with other illicit nodes, so both feature-only and graph models can learn
something. The generator is not meant to mimic Elliptic's statistics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import CLASSES_CSV, EDGES_CSV, FEATURES_CSV, ILLICIT, LICIT, UNKNOWN, TxGraph


def make_synthetic_graph(
    n_steps: int = 12,
    nodes_per_step: int = 80,
    n_local: int = 16,
    n_aggregated: int = 8,
    illicit_rate: float = 0.2,
    unknown_rate: float = 0.4,
    homophily: float = 0.85,
    edges_per_node: int = 2,
    seed: int = 0,
) -> TxGraph:
    rng = np.random.default_rng(seed)
    n = n_steps * nodes_per_step
    time_step = np.repeat(np.arange(1, n_steps + 1), nodes_per_step).astype(np.int16)
    is_illicit = rng.random(n) < illicit_rate

    local = rng.normal(size=(n, n_local)).astype(np.float32)
    local[is_illicit, :4] += 0.8

    src, dst = [], []
    for t in range(1, n_steps + 1):
        nodes = np.flatnonzero(time_step == t)
        bad, good = nodes[is_illicit[nodes]], nodes[~is_illicit[nodes]]
        for v in nodes:
            same, other = (bad, good) if is_illicit[v] else (good, bad)
            for _ in range(edges_per_node):
                pool = same if (rng.random() < homophily and len(same) > 1) or len(other) == 0 else other
                u = int(rng.choice(pool))
                if u != v:
                    src.append(v)
                    dst.append(u)
    edge_index = np.array([src, dst], dtype=np.int64)

    neighbour_sum = np.zeros_like(local)
    degree = np.zeros(n, dtype=np.float32)
    for a, b in ((edge_index[0], edge_index[1]), (edge_index[1], edge_index[0])):
        np.add.at(neighbour_sum, a, local[b])
        np.add.at(degree, a, 1.0)
    neighbour_mean = neighbour_sum / np.maximum(degree, 1.0)[:, None]
    aggregated = neighbour_mean[:, :n_aggregated] + rng.normal(scale=0.5, size=(n, n_aggregated))

    y = np.where(is_illicit, ILLICIT, LICIT).astype(np.int8)
    y[rng.random(n) < unknown_rate] = UNKNOWN
    tx_id = rng.choice(10**9, size=n, replace=False).astype(np.int64)

    return TxGraph(
        x=np.hstack([local, aggregated]).astype(np.float32),
        edge_index=edge_index,
        y=y,
        time_step=time_step,
        tx_id=tx_id,
        n_local=n_local,
    )


def write_elliptic_csvs(graph: TxGraph, raw_dir: Path) -> None:
    """Write ``graph`` in the exact CSV layout of the public Elliptic release."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    features = pd.DataFrame(graph.x)
    features.insert(0, "time_step", graph.time_step)
    features.insert(0, "tx_id", graph.tx_id)
    features.to_csv(raw_dir / FEATURES_CSV, header=False, index=False)

    label_names = {ILLICIT: "1", LICIT: "2", UNKNOWN: "unknown"}
    pd.DataFrame({"txId": graph.tx_id, "class": [label_names[int(v)] for v in graph.y]}).to_csv(
        raw_dir / CLASSES_CSV, index=False
    )
    pd.DataFrame(
        {"txId1": graph.tx_id[graph.edge_index[0]], "txId2": graph.tx_id[graph.edge_index[1]]}
    ).to_csv(raw_dir / EDGES_CSV, index=False)
