"""Transformer model for TinyAlphaZero Phase 1/2."""

from __future__ import annotations

import torch
from torch import nn

from model.encoding import ChessPositionEncoding


class ChessTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 23,
        num_moves: int = 4096,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encode = ChessPositionEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_moves),
        )

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2:
            raise ValueError(f"Expected input shape (B, 69), got {tuple(x.shape)}")

        positions = torch.arange(x.size(1), device=x.device)
        pos = self.pos_encode(positions)  # (69, d_model)
        x = self.tok_embed(x) + pos.unsqueeze(0)

        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))

        policy = self.policy_head(x)
        value_logits = self.value_head(x)

        return policy, value_logits

    @staticmethod
    def get_value(value_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(value_logits, dim=-1)
        return probs[:, 2] - probs[:, 0]

    def predict(self, board_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if board_tensor.dim() == 1:
            board_tensor = board_tensor.unsqueeze(0)

        policy_logits, value_logits = self(board_tensor)
        policy_probs = torch.softmax(policy_logits, dim=-1)
        value = self.get_value(value_logits)
        return policy_probs, value.squeeze(0)
