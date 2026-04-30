"""Full Siamese Drug-Target Affinity model.

This model combines:
  - Drug encoder: GIN-based GNN over molecular graphs
  - Protein encoder: 1D CNN over tokenized amino acid sequences

Fusion (per spec):
  - element-wise product (d * p)  -> [B, D]
  - concatenation of both embeddings [d, p] -> [B, 2D]
  - we build a raw fusion vector: concat([d * p, d, p]) -> [B, 3D]
  - then project to a 512-dim vector (2D when D=256)

Regression head:
  512 -> 1024 -> 512 -> 1 with ReLU and dropout.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from config import Config
from models.drug_encoder import DrugGINEncoder
from models.protein_encoder import ProteinCNNEncoder


def fuse_embeddings_raw(drug_emb: Tensor, protein_emb: Tensor) -> Tensor:
    """Fuse two [B, D] embeddings into a [B, 3D] raw feature vector."""

    if drug_emb.shape != protein_emb.shape:
        raise ValueError(
            f"drug_emb and protein_emb must have same shape, got {tuple(drug_emb.shape)} vs {tuple(protein_emb.shape)}"
        )

    prod = drug_emb * protein_emb
    return torch.cat([prod, drug_emb, protein_emb], dim=-1)


class SiameseDTA(nn.Module):
    """Siamese DTA regression model."""

    def __init__(
        self,
        *,
        drug_encoder: DrugGINEncoder,
        protein_encoder: ProteinCNNEncoder,
        embedding_dim: int = 256,
        dropout: float = 0.1,
        num_tasks: int = 1,
    ) -> None:
        super().__init__()

        self.drug_encoder = drug_encoder
        self.protein_encoder = protein_encoder
        self.embedding_dim = int(embedding_dim)
        self.dropout = float(dropout)
        self.num_tasks = int(num_tasks)

        raw_fused_dim = 3 * self.embedding_dim
        fused_dim = 2 * self.embedding_dim  # 512 when embedding_dim=256

        self.fuse_proj = nn.Sequential(
            nn.Linear(raw_fused_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
        )

        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, self.num_tasks),
        )

    def encode(self, batch: "object") -> Tuple[Tensor, Tensor]:
        """Return (drug_embedding, protein_embedding) for a PyG Batch."""

        # Drug side: expects batch.x, batch.edge_index, batch.batch
        drug_emb = self.drug_encoder(batch.x, batch.edge_index, batch.batch)

        # Protein side: expects batch.protein [B, L]
        protein_tokens = getattr(batch, "protein")
        protein_emb = self.protein_encoder(protein_tokens)

        return drug_emb, protein_emb

    def forward(self, batch: "object") -> Dict[str, Tensor]:
        """Forward pass.

        Returns dict with:
          - pred: [B]
          - drug_emb: [B, D]
          - protein_emb: [B, D]
          - fused: [B, 2D]
        """

        drug_emb, protein_emb = self.encode(batch)
        fused_raw = fuse_embeddings_raw(drug_emb, protein_emb)
        fused = self.fuse_proj(fused_raw)

        pred_all = self.mlp(fused)
        family_id = getattr(batch, "family_id", None)

        if family_id is not None and pred_all.shape[1] > 1:
            # family_id expected as [B, 1] or [B]
            fam = family_id.view(-1, 1).long()
            pred = pred_all.gather(dim=1, index=fam).squeeze(-1)
        else:
            pred = pred_all.squeeze(-1)

        return {
            "pred": pred,
            "drug_emb": drug_emb,
            "protein_emb": protein_emb,
            "fused": fused,
            "fused_raw": fused_raw,
            "pred_all": pred_all,
        }


def build_model_from_config(cfg: Config, *, node_feature_dim: int, num_tasks: int = 1) -> SiameseDTA:
    """Factory to build the full model from Config."""

    drug_enc = DrugGINEncoder(
        in_dim=node_feature_dim,
        hidden_dim=cfg.gin_hidden_dim,
        embedding_dim=cfg.embedding_dim,
        num_layers=cfg.gin_num_layers,
        dropout=cfg.dropout,
    )

    prot_enc = ProteinCNNEncoder(
        vocab_size=cfg.protein_vocab_size,
        token_embed_dim=cfg.protein_embed_dim,
        conv_channels=cfg.protein_conv_channels,
        kernel_size=cfg.protein_conv_kernel_size,
        embedding_dim=cfg.embedding_dim,
        dropout=cfg.dropout,
        padding_idx=cfg.protein_vocab.get("<PAD>", 0),
    )

    return SiameseDTA(
        drug_encoder=drug_enc,
        protein_encoder=prot_enc,
        embedding_dim=cfg.embedding_dim,
        dropout=cfg.dropout,
        num_tasks=num_tasks,
    )
