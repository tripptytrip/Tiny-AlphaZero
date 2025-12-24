"""Pure Python Monte Carlo Tree Search implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
import random
from typing import Dict, Tuple

import chess
import torch

from data.encoding import decode_move, encode_board, encode_move
from mcts.node import MCTSNode


@dataclass
class MCTSConfig:
    num_simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25


def get_temperature(move_number: int, temp_threshold: int = 30) -> float:
    if move_number < temp_threshold:
        return 1.0
    return 0.1


def terminal_value(board: chess.Board) -> float:
    result = board.result()
    if result == "1-0":
        outcome = 1.0
    elif result == "0-1":
        outcome = -1.0
    else:
        outcome = 0.0

    return outcome if board.turn == chess.WHITE else -outcome


class MCTS:
    def __init__(self, model, config: MCTSConfig | None = None, device: torch.device | None = None) -> None:
        self.model = model
        self.config = config or MCTSConfig()
        self.device = device or torch.device("cpu")

    def run(self, board: chess.Board) -> Dict[int, int]:
        root = MCTSNode(move_index=None, state=board.copy())

        if not board.is_game_over():
            self._expand(root, board)
            self._add_dirichlet_noise(root)

        for _ in range(self.config.num_simulations):
            node = root
            scratch = board.copy()
            path = [node]

            while node.is_expanded and node.children:
                node = self._select_child(node)
                move = decode_move(node.move_index, board=scratch)
                scratch.push(move)
                path.append(node)

            if scratch.is_game_over():
                value = terminal_value(scratch)
            else:
                value = self._expand(node, scratch)

            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value
                value = -value

        if self.config.num_simulations == 0:
            return {move: int(child.prior * 1_000_000) for move, child in root.children.items()}

        return {move: child.visit_count for move, child in root.children.items()}

    def select_move(self, board: chess.Board, temperature: float = 0.1) -> int:
        visits = self.run(board)
        if not visits:
            raise ValueError("No legal moves available")

        move_indices, counts = zip(*visits.items())
        counts_tensor = torch.tensor(counts, dtype=torch.float32)

        if temperature <= 0.0:
            return move_indices[int(torch.argmax(counts_tensor))]

        scaled = counts_tensor ** (1.0 / temperature)
        probs = scaled / scaled.sum()
        choice = torch.multinomial(probs, 1).item()
        return move_indices[choice]

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        parent_visits = max(1, node.visit_count)
        scored = []
        best_score = -float("inf")
        for child in node.children.values():
            score = (-child.q_value) + (
                self.config.c_puct
                * child.prior
                * (parent_visits**0.5)
                / (1 + child.visit_count)
            )
            scored.append((score, child))
            if score > best_score:
                best_score = score

        best_children = [child for score, child in scored if score >= best_score - 1e-8]
        return random.choice(best_children)

    def _expand(self, node: MCTSNode, board: chess.Board) -> float:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return terminal_value(board)

        with torch.no_grad():
            tokens = torch.tensor(encode_board(board), dtype=torch.long, device=self.device).unsqueeze(0)
            policy_logits, value_logits = self.model(tokens)
            value = self._value_from_logits(value_logits.squeeze(0)).item()

        legal_indices = [encode_move(move) for move in legal_moves]
        logits = policy_logits.squeeze(0)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[legal_indices] = True
        logits = logits.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)

        for move in legal_moves:
            idx = encode_move(move)
            child_state = board.copy()
            child_state.push(move)
            node.children[idx] = MCTSNode(
                move_index=idx,
                state=child_state,
                parent=node,
                prior=float(probs[idx]),
            )

        return value

    def _add_dirichlet_noise(self, node: MCTSNode) -> None:
        if not node.children or self.config.dirichlet_frac <= 0.0:
            return

        move_indices = list(node.children.keys())
        noise = torch.distributions.Dirichlet(
            torch.full((len(move_indices),), self.config.dirichlet_alpha)
        ).sample()

        for idx, move_index in enumerate(move_indices):
            child = node.children[move_index]
            child.prior = (1 - self.config.dirichlet_frac) * child.prior + self.config.dirichlet_frac * float(noise[idx])

    @staticmethod
    def _value_from_logits(value_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(value_logits, dim=-1)
        return probs[2] - probs[0]
