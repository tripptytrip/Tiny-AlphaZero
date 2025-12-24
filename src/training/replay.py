"""Replay buffer for Phase 2 training."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Tuple

import torch


@dataclass
class ReplayBufferConfig:
    capacity: int = 100_000


class ReplayBuffer:
    def __init__(self, config: ReplayBufferConfig | None = None) -> None:
        self.config = config or ReplayBufferConfig()
        self._buffer: Deque[dict] = deque(maxlen=self.config.capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def push(self, game_data: Iterable[dict]) -> None:
        for sample in game_data:
            self._buffer.append(sample)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch_size > len(self._buffer):
            raise ValueError("Not enough samples in replay buffer")

        batch = random.sample(list(self._buffer), batch_size)
        boards = torch.tensor([s["board"] for s in batch], dtype=torch.long)
        policies = torch.tensor([s["policy"] for s in batch], dtype=torch.float32)
        outcomes = torch.tensor([s["outcome"] + 1 for s in batch], dtype=torch.long)

        return boards, policies, outcomes
