"""Node classifiers: an MLP reference and two message-passing GNNs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv

GNN_KINDS = ("mlp", "gcn", "sage")


class NodeClassifier(nn.Module):
    """Stack of ``num_layers`` hidden layers followed by a linear illicit/licit head.

    ``kind="mlp"`` uses dense layers and ignores the graph, which makes it the
    like-for-like reference for measuring what message passing adds.
    """

    def __init__(
        self,
        kind: str,
        in_dim: int,
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if kind not in GNN_KINDS:
            raise ValueError(f"unknown model kind {kind!r}; expected one of {GNN_KINDS}")
        self.kind = kind
        self.dropout = dropout
        dims = [in_dim] + [hidden] * num_layers
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:], strict=True):
            if kind == "mlp":
                layers.append(nn.Linear(d_in, d_out))
            elif kind == "gcn":
                layers.append(GCNConv(d_in, d_out))
            else:
                layers.append(SAGEConv(d_in, d_out, aggr="mean"))
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(hidden, 2)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h) if self.kind == "mlp" else layer(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x, edge_index))
