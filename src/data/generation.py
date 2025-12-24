"""Random and self-play game generation for TinyAlphaZero."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import chess
import torch

from data.encoding import decode_move, encode_board, encode_move


@dataclass
class GenerationConfig:
    max_game_length: int = 200
    min_game_length: int = 10
    biased_sampling: bool = False
    capture_weight: float = 3.0
    check_weight: float = 2.0


@dataclass
class SelfPlayConfig:
    max_game_length: int = 200
    random_opening_ply: int = 10
    random_opening_prob: float = 0.5
    early_game_temp: float = 1.5
    mid_game_temp: float = 1.0
    late_game_temp: float = 0.3
    early_game_moves: int = 10
    mid_game_moves: int = 15
    num_simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25


def get_outcome(board: chess.Board) -> int:
    """Return +1 for white win, -1 for black win, 0 for draw."""
    result = board.result()
    if result == "1-0":
        return 1
    if result == "0-1":
        return -1
    return 0


def biased_move_selection(
    board: chess.Board,
    capture_weight: float = 3.0,
    check_weight: float = 2.0,
) -> chess.Move:
    """Sample a legal move with simple tactical bias."""
    legal_moves = list(board.legal_moves)
    weights: List[float] = []

    for move in legal_moves:
        weight = 1.0
        if board.is_capture(move):
            weight *= capture_weight
        board.push(move)
        if board.is_check():
            weight *= check_weight
        board.pop()
        weights.append(weight)

    return random.choices(legal_moves, weights=weights, k=1)[0]


def generate_game(config: Optional[GenerationConfig] = None) -> List[dict]:
    """Generate a single self-play game and return a list of position samples."""
    if config is None:
        config = GenerationConfig()

    board = chess.Board()
    positions: List[dict] = []

    while not board.is_game_over() and len(positions) < config.max_game_length:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break

        if config.biased_sampling:
            move = biased_move_selection(
                board,
                capture_weight=config.capture_weight,
                check_weight=config.check_weight,
            )
        else:
            move = random.choice(legal_moves)

        positions.append(
            {
                "board": encode_board(board),
                "move": encode_move(move),
                "legal_moves": [encode_move(m) for m in legal_moves],
                "outcome": None,
                "turn": board.turn,
            }
        )

        board.push(move)

    if len(positions) < config.min_game_length:
        return []

    outcome = get_outcome(board)
    for pos in positions:
        pos_outcome = outcome if pos["turn"] == chess.WHITE else -outcome
        pos["outcome"] = pos_outcome
        del pos["turn"]

    return positions


def generate_games(num_games: int, config: Optional[GenerationConfig] = None) -> List[List[dict]]:
    """Generate multiple games, skipping those that are too short."""
    if config is None:
        config = GenerationConfig()

    games: List[List[dict]] = []
    while len(games) < num_games:
        game = generate_game(config)
        if game:
            games.append(game)

    return games


def aggressive_temperature(move_number: int, config: SelfPlayConfig) -> float:
    if move_number < config.early_game_moves:
        return config.early_game_temp
    if move_number < config.mid_game_moves:
        return config.mid_game_temp
    return config.late_game_temp


def _policy_from_visits(visits: Dict[int, int], num_moves: int = 4096) -> Tuple[List[float], List[int]]:
    policy = [0.0] * num_moves
    if not visits:
        return policy, []

    move_indices = list(visits.keys())
    counts = torch.tensor([visits[idx] for idx in move_indices], dtype=torch.float32)
    probs = counts / counts.sum()
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
    probs = scaled / scaled.sum()
    choice = torch.multinomial(probs, 1).item()
    return move_indices[choice]


def generate_self_play_game(
    model,
    config: Optional[SelfPlayConfig] = None,
    device: Optional[torch.device] = None,
    progress: bool = False,
    log_every: int = 5,
) -> List[dict]:
    from mcts.tree import MCTS, MCTSConfig

    if config is None:
        config = SelfPlayConfig()

    device = device or torch.device("cpu")
    board = chess.Board()
    mcts_config = MCTSConfig(
        num_simulations=config.num_simulations,
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_frac=config.dirichlet_frac,
    )
    mcts = MCTS(model, config=mcts_config, device=device)
    model.eval()

    samples: List[dict] = []
    move_number = 0

    while not board.is_game_over() and move_number < config.max_game_length:
        if progress and move_number % log_every == 0:
            print(f"Self-play move {move_number}...")
        if move_number < config.random_opening_ply and random.random() < config.random_opening_prob:
            move = random.choice(list(board.legal_moves))
            board.push(move)
            move_number += 1
            continue

        visits = mcts.run(board)
        policy, move_indices = _policy_from_visits(visits)
        temperature = aggressive_temperature(move_number, config)
        move_index = _sample_from_policy(move_indices, policy, temperature)
        move = decode_move(move_index, board=board)

        samples.append(
            {
                "board": encode_board(board),
                "policy": policy,
                "move": move_index,
                "outcome": None,
                "turn": board.turn,
            }
        )

        board.push(move)
        move_number += 1

    outcome = get_outcome(board)
    for sample in samples:
        sample_outcome = outcome if sample["turn"] == chess.WHITE else -outcome
        sample["outcome"] = sample_outcome
        del sample["turn"]

    return samples
