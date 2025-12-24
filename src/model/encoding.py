"""Position encoding modules for TinyAlphaZero."""

from __future__ import annotations

import torch
from torch import nn


class ChessPositionEncoding(nn.Module):
    """Factored rank+file positional encoding for chess boards."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.rank_embed = nn.Embedding(8, d_model)
        self.file_embed = nn.Embedding(8, d_model)
        self.flag_embed = nn.Embedding(5, d_model)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.dim() != 1:
            raise ValueError(f"Expected positions shape (S,), got {tuple(positions.shape)}")

        positions = positions.to(torch.long)
        square_mask = positions < 64
        flag_mask = ~square_mask

        out = torch.zeros((positions.numel(), self.rank_embed.embedding_dim), device=positions.device)

        if square_mask.any():
            square_positions = positions[square_mask]
            ranks = square_positions // 8
            files = square_positions % 8
            out[square_mask] = self.rank_embed(ranks) + self.file_embed(files)

        if flag_mask.any():
            flag_positions = positions[flag_mask] - 64
            out[flag_mask] = self.flag_embed(flag_positions)

        return out
