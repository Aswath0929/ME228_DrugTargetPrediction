"""Drug (small molecule) encoder implemented as a GIN-based GNN.

Input:
  - torch_geometric Batch/Data with fields:
      x: node features [num_nodes, num_node_features]
      edge_index: [2, num_edges]

Output:
  - drug embedding tensor [batch_size, embedding_dim]

Architecture (per spec):
  3x GINConv -> global mean pooling -> linear projection -> 256-dim embedding
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

try:
    from torch_geometric.nn import GINConv, global_mean_pool
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torch-geometric is required for the drug encoder. Install torch-geometric."
    ) from e


class DrugGINEncoder(nn.Module):
    """GIN-based drug encoder producing a fixed-size embedding."""

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for layer in range(num_layers):
            layer_in = in_dim if layer == 0 else hidden_dim

            mlp = nn.Sequential(
                nn.Linear(layer_in, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(nn=mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.proj = nn.Linear(hidden_dim, embedding_dim)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """Encode a batch of molecular graphs.

        Args:
            x: Node features [N, in_dim]
            edge_index: Graph connectivity [2, E]
            batch: Batch vector mapping each node to its graph id [N]

        Returns:
            Drug embeddings [B, embedding_dim]
        """

        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = self.act(h)
            h = self.drop(h)

        # Graph-level embedding
        g = global_mean_pool(h, batch)
        g = self.proj(g)
        return g


def infer_node_feature_dim_from_data(data: "object") -> int:
    """Convenience helper to infer x feature size from a PyG Data/Batch."""

    x = getattr(data, "x", None)
    if x is None:
        raise ValueError("data.x is required to infer node feature dim")
    if x.ndim != 2:
        raise ValueError(f"Expected data.x to be 2D, got shape {tuple(x.shape)}")
    return int(x.shape[1])
