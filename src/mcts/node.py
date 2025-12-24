"""MCTS node data structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import chess


@dataclass
class MCTSNode:
    move_index: Optional[int]
    state: Optional[chess.Board] = None
    parent: Optional["MCTSNode"] = None
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        exploration = c_puct * self.prior * (parent_visits**0.5) / (1 + self.visit_count)
        return self.q_value + exploration
