"""Checkpoint pool for league-style self-play."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass
class CheckpointPoolConfig:
    max_size: int = 20


class CheckpointPool:
    def __init__(self, config: CheckpointPoolConfig | None = None) -> None:
        self.config = config or CheckpointPoolConfig()
        self._pool: Deque[str] = deque(maxlen=self.config.max_size)

    def __len__(self) -> int:
        return len(self._pool)

    def add(self, path: str) -> None:
        self._pool.append(path)

    def sample(self) -> Optional[str]:
        if not self._pool:
            return None
        return random.choice(list(self._pool))
