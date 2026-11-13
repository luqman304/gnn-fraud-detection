"""Elliptic Bitcoin dataset: download, preprocessing and the temporal split.

The Elliptic dataset (Weber et al., 2019) is a graph of 203,769 Bitcoin transactions
(nodes) and 234,355 payment flows (directed edges) spread over 49 time steps of roughly
two weeks each. Every node has 165 anonymised features: 93 "local" features describing
the transaction itself and 72 "aggregated" features summarising its one-hop neighbourhood.
About 2% of nodes are labelled illicit, 21% licit and the remaining 77% are unlabelled.
"""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ELLIPTIC_BASE_URL = "https://data.pyg.org/datasets/elliptic"
FEATURES_CSV = "elliptic_txs_features.csv"
EDGES_CSV = "elliptic_txs_edgelist.csv"
CLASSES_CSV = "elliptic_txs_classes.csv"
RAW_FILES = (FEATURES_CSV, EDGES_CSV, CLASSES_CSV)

N_LOCAL_FEATURES = 93
FEATURE_SETS = ("local", "all")

ILLICIT, LICIT, UNKNOWN = 1, 0, -1
LABEL_MAP = {"1": ILLICIT, "2": LICIT, "unknown": UNKNOWN}


@dataclass
class TxGraph:
    """A transaction graph held as plain numpy arrays.

    ``y`` is 1 for illicit, 0 for licit and -1 for unlabelled nodes. ``edge_index`` has
    shape ``(2, E)`` and contains node positions (not transaction ids).
    """

    x: np.ndarray
    edge_index: np.ndarray
    y: np.ndarray
    time_step: np.ndarray
    tx_id: np.ndarray
    n_local: int

    @property
    def num_nodes(self) -> int:
        return int(self.y.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def features(self, feature_set: str) -> np.ndarray:
        if feature_set == "all":
            return self.x
        if feature_set == "local":
            return self.x[:, : self.n_local]
        raise ValueError(f"unknown feature set {feature_set!r}; expected one of {FEATURE_SETS}")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            x=self.x,
            edge_index=self.edge_index,
            y=self.y,
            time_step=self.time_step,
            tx_id=self.tx_id,
            n_local=np.array(self.n_local),
        )

    @classmethod
    def load(cls, path: Path) -> TxGraph:
        with np.load(path) as f:
            return cls(
                x=f["x"],
                edge_index=f["edge_index"],
                y=f["y"],
                time_step=f["time_step"],
                tx_id=f["tx_id"],
                n_local=int(f["n_local"]),
            )

    def summary(self) -> dict[str, int]:
        return {
            "nodes": self.num_nodes,
            "edges": self.num_edges,
            "features": int(self.x.shape[1]),
            "local_features": self.n_local,
            "time_steps": int(np.unique(self.time_step).size),
            "illicit": int((self.y == ILLICIT).sum()),
            "licit": int((self.y == LICIT).sum()),
            "unknown": int((self.y == UNKNOWN).sum()),
            "cross_time_step_edges": int(
                (self.time_step[self.edge_index[0]] != self.time_step[self.edge_index[1]]).sum()
            ),
        }


def download_raw(raw_dir: Path, base_url: str = ELLIPTIC_BASE_URL, force: bool = False) -> list[Path]:
    """Download and extract the three Elliptic CSV files into ``raw_dir``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in RAW_FILES:
        target = raw_dir / name
        if target.exists() and not force:
            paths.append(target)
            continue
        archive = raw_dir / f"{name}.zip"
        print(f"downloading {base_url}/{name}.zip")
        urllib.request.urlretrieve(f"{base_url}/{name}.zip", archive)
        with zipfile.ZipFile(archive) as zf:
            member = next(m for m in zf.namelist() if m.endswith(name))
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
        archive.unlink()
        paths.append(target)
    return paths


def build_graph(raw_dir: Path, n_local: int = N_LOCAL_FEATURES) -> TxGraph:
    """Parse the raw CSV files into a :class:`TxGraph`.

    The features file has no header: column 0 is the transaction id, column 1 the time
    step and the remaining columns are the anonymised features.
    """
    feats = pd.read_csv(raw_dir / FEATURES_CSV, header=None)
    tx_id = feats[0].to_numpy(np.int64)
    time_step = feats[1].to_numpy(np.int16)
    x = feats.iloc[:, 2:].to_numpy(np.float32)
    del feats
    if len(np.unique(tx_id)) != len(tx_id):
        raise ValueError("duplicate transaction ids in features file")

    classes = pd.read_csv(raw_dir / CLASSES_CSV, dtype={"txId": np.int64, "class": str})
    labels = classes.set_index("txId")["class"].map(LABEL_MAP).reindex(tx_id)
    if labels.isna().any():
        raise ValueError(f"{int(labels.isna().sum())} transactions have no class entry")
    y = labels.to_numpy(np.int8)

    edges = pd.read_csv(raw_dir / EDGES_CSV, dtype=np.int64)
    position = pd.Series(np.arange(len(tx_id), dtype=np.int64), index=tx_id)
    src = position.reindex(edges.iloc[:, 0])
    dst = position.reindex(edges.iloc[:, 1])
    if src.isna().any() or dst.isna().any():
        raise ValueError("edge list references unknown transaction ids")
    edge_index = np.vstack([src.to_numpy(np.int64), dst.to_numpy(np.int64)])

    return TxGraph(x=x, edge_index=edge_index, y=y, time_step=time_step, tx_id=tx_id, n_local=n_local)


@dataclass(frozen=True)
class TemporalSplit:
    """Split nodes by time step so that no model is trained on the future.

    Steps ``1..train_end`` are used for fitting, ``train_end+1..val_end`` for early
    stopping and model selection, and everything after ``val_end`` is the test set.
    With the defaults the final models are refit on steps 1-34 and tested on 35-49,
    matching the 34/15 split of Weber et al. (2019).
    """

    train_end: int = 29
    val_end: int = 34

    def masks(self, graph: TxGraph) -> dict[str, np.ndarray]:
        labelled = graph.y != UNKNOWN
        ts = graph.time_step
        train = ts <= self.train_end
        val = (ts > self.train_end) & (ts <= self.val_end)
        test = ts > self.val_end
        return {
            "train": train & labelled,
            "val": val & labelled,
            "fit": (train | val) & labelled,
            "test": test & labelled,
        }
