"""Loss functions for DTA regression.

Combined objective (per spec):
  L = MSE(y_pred, y_true) + lambda * contrastive_loss

Contrastive term:
  For *non-binder* pairs, penalize cosine similarity between drug and protein
  embeddings when it is too high.

We use a hinge-style penalty:
  contrast = mean( relu(cos_sim - max_cosine)^2 ) over non-binders

`is_nonbinder` can be provided by the dataset; otherwise we can infer it from
(y_true, cfg.dataset) thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from config import Config


def infer_nonbinder_mask(y_true: Tensor, cfg: Config) -> Tensor:
    """Infer non-binder mask from labels using dataset-specific thresholds."""

    y = y_true.view(-1)
    if cfg.dataset == "DAVIS":
        return y < cfg.davis_nonbinder_pkd_threshold

    # KIBA: per user spec
    return y > cfg.kiba_nonbinder_score_threshold


class CombinedLoss(nn.Module):
    """MSE + contrastive penalty for non-binders."""

    def __init__(
        self,
        *,
        cfg: Config,
        mse_weight: Optional[float] = None,
        contrastive_weight: Optional[float] = None,
        nonbinder_max_cosine: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.mse_weight = float(cfg.mse_weight if mse_weight is None else mse_weight)
        self.contrastive_weight = float(
            cfg.contrastive_weight if contrastive_weight is None else contrastive_weight
        )
        self.nonbinder_max_cosine = float(
            cfg.nonbinder_max_cosine if nonbinder_max_cosine is None else nonbinder_max_cosine
        )

    def forward(
        self,
        *,
        pred: Tensor,
        y_true: Tensor,
        drug_emb: Tensor,
        protein_emb: Tensor,
        is_nonbinder: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute combined loss.

        Args:
            pred: [B] or [B, 1]
            y_true: [B] or [B, 1]
            drug_emb: [B, D]
            protein_emb: [B, D]
            is_nonbinder: optional bool mask, shape [B] or [B, 1]

        Returns:
            Scalar loss tensor.
        """

        pred = pred.view(-1)
        y_true = y_true.view(-1)

        mse = F.mse_loss(pred, y_true)

        if is_nonbinder is None:
            mask = infer_nonbinder_mask(y_true, self.cfg)
        else:
            mask = is_nonbinder.view(-1).bool()

        # Contrastive penalty only applies to non-binders.
        if mask.any():
            # Cosine similarity per sample: [B]
            cos = F.cosine_similarity(drug_emb, protein_emb, dim=-1)
            hinge = F.relu(cos - self.nonbinder_max_cosine)
            contrastive = (hinge * hinge)[mask].mean()
        else:
            contrastive = torch.zeros((), device=pred.device, dtype=pred.dtype)

        return self.mse_weight * mse + self.contrastive_weight * contrastive
