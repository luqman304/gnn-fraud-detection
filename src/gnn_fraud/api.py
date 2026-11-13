"""FastAPI service that scores Elliptic transactions and their neighbourhood.

On start-up the service loads the processed graph and a trained model, then scores
every node in one full-graph pass. Requests are answered from that in-memory snapshot,
so latency does not depend on the model type. This matches how the dataset is used:
the graph is a fixed batch of historical transactions and the model is transductive.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from scipy import sparse

from . import __version__
from .data import ILLICIT, LICIT, TxGraph
from .training import TrainedModel


class NodeScore(BaseModel):
    tx_id: int
    time_step: int
    illicit_probability: float
    predicted_label: str
    known_label: str = Field(description="Ground-truth label in the dataset: illicit, licit or unknown")


class Neighbour(NodeScore):
    hop: int


class TransactionReport(NodeScore):
    threshold: float
    degree: int
    neighbourhood: list[Neighbour]
    neighbourhood_size: int
    neighbours_flagged: int
    neighbours_known_illicit: int


class BatchRequest(BaseModel):
    tx_ids: list[int] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    scores: list[NodeScore]
    missing: list[int]


class ModelCard(BaseModel):
    kind: str
    feature_set: str
    threshold: float
    info: dict
    nodes_scored: int
    flagged: int


@dataclass
class ScoringState:
    graph: TxGraph
    model: TrainedModel
    prob: np.ndarray
    position: dict[int, int]
    adjacency: sparse.csr_matrix

    @classmethod
    def build(cls, graph: TxGraph, model: TrainedModel) -> ScoringState:
        n = graph.num_nodes
        src, dst = graph.edge_index
        adj = sparse.coo_matrix((np.ones(src.size, dtype=np.int8), (src, dst)), shape=(n, n)).tocsr()
        adj = ((adj + adj.T) > 0).astype(np.int8).tocsr()
        return cls(
            graph=graph,
            model=model,
            prob=model.predict_proba(graph),
            position={int(t): i for i, t in enumerate(graph.tx_id)},
            adjacency=adj,
        )

    def _label(self, value: int) -> str:
        return {ILLICIT: "illicit", LICIT: "licit"}.get(int(value), "unknown")

    def node(self, i: int) -> dict:
        p = float(self.prob[i])
        return {
            "tx_id": int(self.graph.tx_id[i]),
            "time_step": int(self.graph.time_step[i]),
            "illicit_probability": round(p, 6),
            "predicted_label": "illicit" if p >= self.model.threshold else "licit",
            "known_label": self._label(self.graph.y[i]),
        }

    def neighbourhood(self, i: int, hops: int) -> list[tuple[int, int]]:
        """Breadth-first search; returns ``(node, hop)`` pairs excluding the root."""
        seen = {i}
        frontier = [i]
        found: list[tuple[int, int]] = []
        for hop in range(1, hops + 1):
            nxt = []
            for v in frontier:
                for u in self.adjacency.indices[self.adjacency.indptr[v] : self.adjacency.indptr[v + 1]]:
                    if u not in seen:
                        seen.add(int(u))
                        nxt.append(int(u))
                        found.append((int(u), hop))
            frontier = nxt
        return found


def create_app(graph_path: Path, artifact_dir: Path) -> FastAPI:
    state: dict[str, ScoringState] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["s"] = ScoringState.build(TxGraph.load(graph_path), TrainedModel.load(artifact_dir))
        yield
        state.clear()

    app = FastAPI(
        title="Elliptic illicit-transaction scoring",
        version=__version__,
        lifespan=lifespan,
    )

    def scoring() -> ScoringState:
        if "s" not in state:
            raise HTTPException(503, "model not loaded")
        return state["s"]

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok" if "s" in state else "loading"}

    @app.get("/model", response_model=ModelCard)
    def model_card() -> ModelCard:
        s = scoring()
        return ModelCard(
            kind=s.model.kind,
            feature_set=s.model.feature_set,
            threshold=s.model.threshold,
            info=s.model.info,
            nodes_scored=int(s.prob.size),
            flagged=int((s.prob >= s.model.threshold).sum()),
        )

    @app.get("/transactions/{tx_id}", response_model=TransactionReport)
    def score_transaction(
        tx_id: int,
        hops: int = Query(1, ge=0, le=3, description="Neighbourhood radius in hops"),
        max_neighbours: int = Query(50, ge=0, le=1000, description="Return at most this many neighbours"),
    ) -> TransactionReport:
        s = scoring()
        i = s.position.get(tx_id)
        if i is None:
            raise HTTPException(404, f"transaction {tx_id} is not in the graph")
        found = s.neighbourhood(i, hops)
        found.sort(key=lambda item: (item[1], -s.prob[item[0]]))
        idx = np.array([u for u, _ in found], dtype=np.int64)
        return TransactionReport(
            **s.node(i),
            threshold=s.model.threshold,
            degree=int(s.adjacency.indptr[i + 1] - s.adjacency.indptr[i]),
            neighbourhood=[Neighbour(**s.node(u), hop=h) for u, h in found[:max_neighbours]],
            neighbourhood_size=len(found),
            neighbours_flagged=int((s.prob[idx] >= s.model.threshold).sum()) if idx.size else 0,
            neighbours_known_illicit=int((s.graph.y[idx] == ILLICIT).sum()) if idx.size else 0,
        )

    @app.post("/score", response_model=BatchResponse)
    def score_batch(req: BatchRequest) -> BatchResponse:
        s = scoring()
        scores, missing = [], []
        for tx_id in req.tx_ids:
            i = s.position.get(tx_id)
            if i is None:
                missing.append(tx_id)
            else:
                scores.append(NodeScore(**s.node(i)))
        return BatchResponse(scores=scores, missing=missing)

    return app


def app_from_env() -> FastAPI:
    """Uvicorn factory: ``uvicorn gnn_fraud.api:app_from_env --factory``."""
    import os

    from .config import get_paths

    paths = get_paths()
    artifact = Path(os.environ.get("GNNFRAUD_ARTIFACT", paths.default_artifact))
    return create_app(paths.graph_file, artifact)
