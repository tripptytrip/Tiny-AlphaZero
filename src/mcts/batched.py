"""Vectorized (virtual) batched MCTS implementation."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import chess
import torch

from data.encoding import decode_move, encode_board, encode_move
from mcts.node import MCTSNode


@dataclass
class BatchedMCTSConfig:
    num_simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25


def terminal_value(board: chess.Board) -> float:
    result = board.result()
    if result == "1-0":
        outcome = 1.0
    elif result == "0-1":
        outcome = -1.0
    else:
        outcome = 0.0

    return outcome if board.turn == chess.WHITE else -outcome


def _policy_from_visits(visits: Dict[int, int], num_moves: int = 4096) -> Tuple[List[float], List[int]]:
    policy = [0.0] * num_moves
    if not visits:
        return policy, []

    move_indices = list(visits.keys())
    counts = torch.tensor([visits[idx] for idx in move_indices], dtype=torch.float32)
    total = counts.sum()
    if total <= 0:
        probs = torch.full_like(counts, 1.0 / len(move_indices))
    else:
        probs = counts / total
    for idx, prob in zip(move_indices, probs.tolist()):
        policy[idx] = prob
    return policy, move_indices


def _sample_from_policy(move_indices: Sequence[int], policy: Sequence[float], temperature: float) -> int:
    if not move_indices:
        raise ValueError("No moves to sample from")

    if temperature <= 0.0:
        return max(move_indices, key=lambda idx: policy[idx])

    logits = torch.tensor([policy[idx] for idx in move_indices], dtype=torch.float32)
    scaled = logits ** (1.0 / temperature)
    total = scaled.sum()
    if total <= 0:
        probs = torch.full_like(scaled, 1.0 / len(move_indices))
    else:
        probs = scaled / total
    choice = torch.multinomial(probs, 1).item()
    return move_indices[choice]


def aggressive_temperature(
    move_number: int,
    early_game_moves: int = 10,
    mid_game_moves: int = 15,
    early_game_temp: float = 1.5,
    mid_game_temp: float = 1.0,
    late_game_temp: float = 0.3,
) -> float:
    if move_number < early_game_moves:
        return early_game_temp
    if move_number < mid_game_moves:
        return mid_game_temp
    return late_game_temp


class BatchedMCTS:
    def __init__(
        self,
        model,
        config: BatchedMCTSConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config or BatchedMCTSConfig()
        self.device = device or torch.device("cpu")

    def run_batch(self, boards: List[chess.Board]) -> List[Dict[int, int]]:
        roots = [MCTSNode(move_index=None, state=b.copy()) for b in boards]
        start = time.perf_counter()

        for root in roots:
            if root.state is not None and not root.state.is_game_over():
                self._expand_root(root)
                self._add_dirichlet_noise(root)

        for _ in range(self.config.num_simulations):
            leaves: List[MCTSNode] = []
            eval_nodes: List[MCTSNode] = []

            for root in roots:
                node = root
                while node.is_expanded and node.children:
                    node = self._select_child(node)
                leaves.append(node)
                if node.state is not None and not node.state.is_game_over():
                    eval_nodes.append(node)

            values: Dict[int, float] = {}
            if eval_nodes:
                batch_tokens = torch.stack(
                    [
                        torch.tensor(encode_board(node.state), dtype=torch.long)
                        for node in eval_nodes
                    ]
                ).to(self.device)

                with torch.no_grad():
                    policy_logits, value_logits = self.model(batch_tokens)

                for idx, node in enumerate(eval_nodes):
                    board = node.state
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        values[id(node)] = terminal_value(board)
                        continue

                    logits = policy_logits[idx]
                    legal_indices = [encode_move(move) for move in legal_moves]
                    mask = torch.zeros_like(logits, dtype=torch.bool)
                    mask[legal_indices] = True
                    masked = logits.masked_fill(~mask, float("-inf"))
                    probs = torch.softmax(masked, dim=-1)

                    for move in legal_moves:
                        move_idx = encode_move(move)
                        child_state = board.copy()
                        child_state.push(move)
                        node.children[move_idx] = MCTSNode(
                            move_index=move_idx,
                            state=child_state,
                            parent=node,
                            prior=float(probs[move_idx]),
                        )

                    values[id(node)] = self._value_from_logits(value_logits[idx]).item()

            for node in leaves:
                if node.state is None:
                    continue
                if node.state.is_game_over():
                    value = terminal_value(node.state)
                else:
                    value = values.get(id(node), 0.0)

                current = node
                while current is not None:
                    current.visit_count += 1
                    current.value_sum += value
                    value = -value
                    current = current.parent

        self._last_duration = time.perf_counter() - start

        return [
            {move: child.visit_count for move, child in root.children.items()} for root in roots
        ]

    @property
    def last_duration(self) -> float:
        return getattr(self, "_last_duration", 0.0)

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

    @staticmethod
    def _value_from_logits(value_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(value_logits, dim=-1)
        return probs[2] - probs[0]

    def _expand_root(self, node: MCTSNode) -> None:
        board = node.state
        if board is None:
            return
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return

        with torch.no_grad():
            tokens = torch.tensor(encode_board(board), dtype=torch.long, device=self.device).unsqueeze(0)
            policy_logits, _ = self.model(tokens)
        logits = policy_logits.squeeze(0)
        legal_indices = [encode_move(move) for move in legal_moves]
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[legal_indices] = True
        masked = logits.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(masked, dim=-1)

        for move in legal_moves:
            move_idx = encode_move(move)
            child_state = board.copy()
            child_state.push(move)
            node.children[move_idx] = MCTSNode(
                move_index=move_idx,
                state=child_state,
                parent=node,
                prior=float(probs[move_idx]),
            )

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


def generate_self_play_batch(
    model,
    batch_size: int,
    num_simulations: int,
    max_game_length: int,
    device: torch.device | None = None,
    dirichlet_alpha: float = 0.3,
    dirichlet_frac: float = 0.25,
    random_opening_ply: int = 10,
    random_opening_prob: float = 0.5,
) -> List[List[dict]]:
    device = device or torch.device("cpu")
    model.eval()
    mcts = BatchedMCTS(
        model,
        config=BatchedMCTSConfig(
            num_simulations=num_simulations,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_frac=dirichlet_frac,
        ),
        device=device,
    )
    start = time.perf_counter()

    boards = [chess.Board() for _ in range(batch_size)]
    samples: List[List[dict]] = [[] for _ in range(batch_size)]
    move_numbers = [0 for _ in range(batch_size)]
    finished = [False for _ in range(batch_size)]

    while not all(finished):
        active_indices = [i for i, done in enumerate(finished) if not done]
        active_boards = [boards[i] for i in active_indices]

        random_indices = []
        mcts_indices = []
        if random_opening_ply > 0:
            for local_idx, board_idx in enumerate(active_indices):
                if move_numbers[board_idx] < random_opening_ply and random.random() < random_opening_prob:
                    random_indices.append((local_idx, board_idx))
                else:
                    mcts_indices.append((local_idx, board_idx))
        else:
            mcts_indices = list(enumerate(active_indices))

        visit_batches = []
        if mcts_indices:
            mcts_boards = [boards[idx] for _, idx in mcts_indices]
            visit_batches = mcts.run_batch(mcts_boards)

        mcts_cursor = 0
        random_set = {idx for _, idx in random_indices}
        for local_idx, board_idx in enumerate(active_indices):
            board = boards[board_idx]
            if not board.is_game_over() and move_numbers[board_idx] < max_game_length:
                is_random = board_idx in random_set

                if is_random:
                    move = random.choice(list(board.legal_moves))
                    board.push(move)
                    move_numbers[board_idx] += 1
                else:
                    visits = visit_batches[mcts_cursor]
                    mcts_cursor += 1
                    policy, move_indices = _policy_from_visits(visits)

                    if not move_indices:
                        move = random.choice(list(board.legal_moves))
                        move_index = encode_move(move)
                        policy = [0.0] * 4096
                        policy[move_index] = 1.0
                    else:
                        temperature = aggressive_temperature(move_numbers[board_idx])
                        move_index = _sample_from_policy(move_indices, policy, temperature)
                        move = decode_move(move_index, board=board)

                    samples[board_idx].append(
                        {
                            "board": encode_board(board),
                            "policy": policy,
                            "move": move_index,
                            "outcome": None,
                            "turn": board.turn,
                        }
                    )

                    board.push(move)
                    move_numbers[board_idx] += 1

            if board.is_game_over() or move_numbers[board_idx] >= max_game_length:
                finished[board_idx] = True

    for idx, board in enumerate(boards):
        outcome = 0
        result = board.result()
        if result == "1-0":
            outcome = 1
        elif result == "0-1":
            outcome = -1

        for sample in samples[idx]:
            sample_outcome = outcome if sample["turn"] == chess.WHITE else -outcome
            sample["outcome"] = sample_outcome
            del sample["turn"]

    total_positions = sum(len(game) for game in samples)
    duration = max(1e-8, time.perf_counter() - start)
    print(f"Batched self-play: {total_positions / duration:.2f} positions/sec", flush=True)

    return samples
