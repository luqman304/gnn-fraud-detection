"""Training under the temporal protocol, and persistence of trained models.

Every model sees exactly the same labelled nodes:

* feature baselines (logistic regression, random forest) are fit on the labelled nodes
  of time steps ``1..val_end``;
* neural models are first trained on ``1..train_end`` with early stopping on the
  validation PR-AUC of steps ``train_end+1..val_end``; they are then retrained from
  scratch on ``1..val_end`` for the number of epochs selected in the first stage.

No test-set label or statistic is used at any point, including feature scaling.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_undirected

from .data import TemporalSplit, TxGraph
from .models import GNN_KINDS, NodeClassifier

MODEL_KINDS = ("lr", "rf", "mlp", "gcn", "sage", "rf+sage")


@dataclass
class GNNConfig:
    hidden: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    lr: float = 0.01
    weight_decay: float = 5e-4
    max_epochs: int = 200
    patience: int = 30
    illicit_weight: float = 0.7
    device: str = "cpu"


@dataclass
class RFConfig:
    n_estimators: int = 100
    max_features: int = 50


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> Standardizer:
        std = x.std(axis=0)
        return cls(mean=x.mean(axis=0), std=np.where(std > 0, std, 1.0).astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device: str) -> torch.device:
    """``auto`` picks CUDA, then Apple MPS, then CPU."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _edge_tensor(graph: TxGraph) -> torch.Tensor:
    return to_undirected(torch.from_numpy(graph.edge_index), num_nodes=graph.num_nodes)


@dataclass
class TrainedModel:
    """A fitted model that scores every node of a :class:`TxGraph`."""

    kind: str
    feature_set: str
    estimator: Any
    scaler: Standardizer | None = None
    gnn_config: GNNConfig | None = None
    embedder: TrainedModel | None = None
    threshold: float = 0.5
    info: dict[str, Any] = field(default_factory=dict)

    def _gnn_inputs(self, graph: TxGraph) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.scaler is not None
        x = torch.from_numpy(self.scaler.transform(graph.features(self.feature_set)))
        return x, _edge_tensor(graph)

    def _sklearn_inputs(self, graph: TxGraph) -> np.ndarray:
        x = graph.features(self.feature_set)
        if self.embedder is not None:
            x = np.hstack([x, self.embedder.embed(graph)])
        return x

    @torch.no_grad()
    def embed(self, graph: TxGraph) -> np.ndarray:
        if self.kind not in GNN_KINDS:
            raise ValueError(f"{self.kind} has no node embeddings")
        self.estimator.eval()
        return self.estimator.embed(*self._gnn_inputs(graph)).numpy()

    @torch.no_grad()
    def predict_proba(self, graph: TxGraph) -> np.ndarray:
        """Probability that each node is illicit, for every node in ``graph``."""
        if self.kind in GNN_KINDS:
            self.estimator.eval()
            logits = self.estimator(*self._gnn_inputs(graph))
            return torch.softmax(logits, dim=1)[:, 1].numpy()
        return self.estimator.predict_proba(self._sklearn_inputs(graph))[:, 1]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "kind": self.kind,
            "feature_set": self.feature_set,
            "threshold": self.threshold,
            "gnn_config": asdict(self.gnn_config) if self.gnn_config else None,
            "info": self.info,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))
        if self.scaler is not None:
            np.savez(directory / "scaler.npz", mean=self.scaler.mean, std=self.scaler.std)
        if self.kind in GNN_KINDS:
            torch.save(self.estimator.state_dict(), directory / "model.pt")
        else:
            joblib.dump(self.estimator, directory / "model.joblib")
        if self.embedder is not None:
            self.embedder.save(directory / "embedder")

    @classmethod
    def load(cls, directory: Path) -> TrainedModel:
        meta = json.loads((directory / "meta.json").read_text())
        scaler = None
        if (directory / "scaler.npz").exists():
            with np.load(directory / "scaler.npz") as f:
                scaler = Standardizer(mean=f["mean"], std=f["std"])
        gnn_config = GNNConfig(**meta["gnn_config"]) if meta["gnn_config"] else None
        if meta["kind"] in GNN_KINDS:
            assert gnn_config is not None and scaler is not None
            model = NodeClassifier(
                meta["kind"],
                scaler.mean.shape[0],
                gnn_config.hidden,
                gnn_config.num_layers,
                gnn_config.dropout,
            )
            model.load_state_dict(torch.load(directory / "model.pt", weights_only=True))
            estimator: Any = model
        else:
            estimator = joblib.load(directory / "model.joblib")
        embedder = cls.load(directory / "embedder") if (directory / "embedder").exists() else None
        return cls(
            kind=meta["kind"],
            feature_set=meta["feature_set"],
            estimator=estimator,
            scaler=scaler,
            gnn_config=gnn_config,
            embedder=embedder,
            threshold=meta["threshold"],
            info=meta["info"],
        )


def _fit_network(
    kind: str,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor | None,
    cfg: GNNConfig,
    seed: int,
    epochs: int | None = None,
) -> tuple[NodeClassifier, int, float | None]:
    """Full-batch training. With ``val_mask`` returns the best epoch by validation PR-AUC.

    Training runs on ``cfg.device``; the returned model is always on the CPU.
    """
    set_seed(seed)
    device = resolve_device(cfg.device)
    x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
    train_mask = train_mask.to(device)
    model = NodeClassifier(kind, x.shape[1], cfg.hidden, cfg.num_layers, cfg.dropout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    class_weight = torch.tensor([1.0 - cfg.illicit_weight, cfg.illicit_weight], device=device)
    if val_mask is not None:
        val_mask = val_mask.to(device)
        y_val = y[val_mask].cpu().numpy()

    best_score, best_epoch, best_state = -np.inf, 0, None
    for epoch in range(1, (epochs or cfg.max_epochs) + 1):
        model.train()
        optimiser.zero_grad()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=class_weight)
        loss.backward()
        optimiser.step()

        if val_mask is None:
            continue
        model.eval()
        with torch.no_grad():
            prob = torch.softmax(model(x, edge_index)[val_mask], dim=1)[:, 1].cpu().numpy()
        score = average_precision_score(y_val, prob)
        if score > best_score:
            best_score, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            break

    if val_mask is None:
        return model.cpu(), epochs or cfg.max_epochs, None
    assert best_state is not None
    model.load_state_dict(best_state)
    return model.cpu(), best_epoch, float(best_score)


def train_gnn(
    kind: str,
    graph: TxGraph,
    split: TemporalSplit,
    feature_set: str,
    seed: int,
    cfg: GNNConfig | None = None,
    refit: bool = True,
) -> TrainedModel:
    cfg = cfg or GNNConfig()
    masks = {k: torch.from_numpy(v) for k, v in split.masks(graph).items()}
    y = torch.from_numpy(np.clip(graph.y, 0, 1).astype(np.int64))
    edge_index = _edge_tensor(graph)
    features = graph.features(feature_set)

    scaler = Standardizer.fit(features[graph.time_step <= split.train_end])
    x = torch.from_numpy(scaler.transform(features))
    model, best_epoch, val_pr_auc = _fit_network(
        kind, x, edge_index, y, masks["train"], masks["val"], cfg, seed
    )
    info = {"seed": seed, "best_epoch": best_epoch, "val_pr_auc": val_pr_auc, "refit": refit}

    if refit:
        scaler = Standardizer.fit(features[graph.time_step <= split.val_end])
        x = torch.from_numpy(scaler.transform(features))
        model, _, _ = _fit_network(kind, x, edge_index, y, masks["fit"], None, cfg, seed, epochs=best_epoch)
        info["trained_on_steps"] = [1, split.val_end]
    else:
        info["trained_on_steps"] = [1, split.train_end]

    return TrainedModel(
        kind=kind, feature_set=feature_set, estimator=model, scaler=scaler, gnn_config=cfg, info=info
    )


def _sklearn_estimator(kind: str, seed: int, n_features: int, rf_cfg: RFConfig) -> Any:
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    return RandomForestClassifier(
        n_estimators=rf_cfg.n_estimators,
        max_features=min(rf_cfg.max_features, n_features),
        n_jobs=-1,
        random_state=seed,
    )


def train_model(
    kind: str,
    graph: TxGraph,
    split: TemporalSplit,
    feature_set: str = "all",
    seed: int = 0,
    gnn_cfg: GNNConfig | None = None,
    rf_cfg: RFConfig | None = None,
    embedder: TrainedModel | None = None,
) -> TrainedModel:
    """Train one model of ``kind`` (see :data:`MODEL_KINDS`) under the temporal protocol.

    For ``rf+sage`` an already trained GraphSAGE model can be passed as ``embedder``
    to avoid training it a second time.
    """
    if kind not in MODEL_KINDS:
        raise ValueError(f"unknown model kind {kind!r}; expected one of {MODEL_KINDS}")
    if kind in GNN_KINDS:
        return train_gnn(kind, graph, split, feature_set, seed, gnn_cfg)

    rf_cfg = rf_cfg or RFConfig()
    fit = split.masks(graph)["fit"]
    x = graph.features(feature_set)
    if kind == "rf+sage":
        if embedder is None:
            embedder = train_gnn("sage", graph, split, feature_set, seed, gnn_cfg)
        elif embedder.kind != "sage" or embedder.feature_set != feature_set:
            raise ValueError("embedder must be a GraphSAGE model trained on the same feature set")
        x = np.hstack([x, embedder.embed(graph)])
    else:
        embedder = None
    estimator = _sklearn_estimator(kind.split("+")[0], seed, x.shape[1], rf_cfg)
    estimator.fit(x[fit], graph.y[fit])
    return TrainedModel(
        kind=kind,
        feature_set=feature_set,
        estimator=estimator,
        embedder=embedder,
        info={"seed": seed, "trained_on_steps": [1, split.val_end]},
    )
