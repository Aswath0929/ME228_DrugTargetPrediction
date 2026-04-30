"""Protein encoder implemented as a 1D CNN over tokenized amino acid sequences.

Input:
  - protein token ids: LongTensor [batch_size, max_len]

Output:
  - protein embedding: FloatTensor [batch_size, embedding_dim]

Architecture (per spec):
  Embedding(25 -> 128)
  -> Conv1d stacks with channels [32, 64, 96], kernel size 8
  -> global max pooling over sequence length
  -> Linear -> 256-dim embedding
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class ProteinCNNEncoder(nn.Module):
    """1D CNN protein encoder producing a fixed-size embedding."""

    def __init__(
        self,
        *,
        vocab_size: int = 25,
        token_embed_dim: int = 128,
        conv_channels: Sequence[int] = (32, 64, 96),
        kernel_size: int = 8,
        embedding_dim: int = 256,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if len(conv_channels) != 3:
            raise ValueError("conv_channels must have length 3 (filters: 32, 64, 96)")

        self.vocab_size = vocab_size
        self.token_embed_dim = token_embed_dim
        self.conv_channels = tuple(int(c) for c in conv_channels)
        self.kernel_size = int(kernel_size)
        self.embedding_dim = int(embedding_dim)
        self.dropout = float(dropout)
        self.padding_idx = int(padding_idx)

        self.embed = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.token_embed_dim,
            padding_idx=self.padding_idx,
        )

        c1, c2, c3 = self.conv_channels

        # Use padding='same' behavior manually for stable sequence length.
        # With kernel_size=8, pad=(k//2)=4 makes output length ~input length.
        pad = self.kernel_size // 2

        self.conv1 = nn.Conv1d(self.token_embed_dim, c1, kernel_size=self.kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=self.kernel_size, padding=pad)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=self.kernel_size, padding=pad)

        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=self.dropout)

        self.proj = nn.Linear(c3, self.embedding_dim)

    def forward(self, protein_tokens: Tensor) -> Tensor:
        """Encode protein sequences.

        Args:
            protein_tokens: LongTensor [B, L] (preferred) or [L]

        Returns:
            Protein embeddings [B, embedding_dim]
        """

        if protein_tokens.ndim == 1:
            protein_tokens = protein_tokens.unsqueeze(0)

        if protein_tokens.dtype != torch.long:
            protein_tokens = protein_tokens.long()

        # [B, L, E]
        x = self.embed(protein_tokens)
        x = self.drop(x)

        # Conv1d expects [B, C, L]
        x = x.transpose(1, 2)

        x = self.act(self.conv1(x))
        x = self.drop(x)
        x = self.act(self.conv2(x))
        x = self.drop(x)
        x = self.act(self.conv3(x))

        # Global max pool over length -> [B, C]
        x = torch.amax(x, dim=-1)

        x = self.proj(x)
        return x
