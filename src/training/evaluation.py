"""Evaluation utilities for Phase 2."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import chess
import torch

from mcts.tree import MCTS, MCTSConfig
from data.encoding import decode_move


@dataclass
class EloConfig:
    k_factor: float = 32.0


class CheckpointPool:
    def __init__(self, max_size: int = 20) -> None:
        self._pool: Deque[str] = deque(maxlen=max_size)

    def __len__(self) -> int:
        return len(self._pool)

    def add(self, checkpoint_path: str) -> None:
        self._pool.append(checkpoint_path)

    def sample_opponent(self, exclude: Optional[str] = None) -> Optional[str]:
        candidates = [path for path in self._pool if path != exclude]
        if not candidates:
            return None
        return random.choice(candidates)


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, score_a: float, k_factor: float) -> float:
    return rating_a + k_factor * (score_a - expected_score(rating_a, rating_b))


def play_game(
    model_white,
    model_black,
    num_simulations: int,
    device: torch.device,
    max_moves: int = 200,
) -> int:
    board = chess.Board()
    mcts_white = MCTS(model_white, config=MCTSConfig(num_simulations=num_simulations), device=device)
    mcts_black = MCTS(model_black, config=MCTSConfig(num_simulations=num_simulations), device=device)

    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        if board.turn == chess.WHITE:
            move_index = mcts_white.select_move(board, temperature=0.0)
        else:
            move_index = mcts_black.select_move(board, temperature=0.0)

        move = decode_move(move_index, board=board)
        board.push(move)
        move_count += 1

    result = board.result()
    if result == "1-0":
        return 1
    if result == "0-1":
        return -1
    return 0


def evaluate_model(
    challenger,
    incumbent,
    num_games: int = 10,
    num_simulations: int = 50,
    device: Optional[torch.device] = None,
    elo_config: Optional[EloConfig] = None,
) -> Dict[str, float]:
    device = device or torch.device("cpu")
    elo_config = elo_config or EloConfig()

    challenger.eval()
    incumbent.eval()

    challenger_score = 0.0
    incumbent_score = 0.0

    for game_idx in range(num_games):
        if game_idx % 2 == 0:
            result = play_game(challenger, incumbent, num_simulations, device)
            if result == 1:
                challenger_score += 1.0
            elif result == -1:
                incumbent_score += 1.0
            else:
                challenger_score += 0.5
                incumbent_score += 0.5
        else:
            result = play_game(incumbent, challenger, num_simulations, device)
            if result == 1:
                incumbent_score += 1.0
            elif result == -1:
                challenger_score += 1.0
            else:
                challenger_score += 0.5
                incumbent_score += 0.5

    rating_challenger = 1000.0
    rating_incumbent = 1000.0
    score_a = challenger_score / max(1.0, num_games)
    rating_challenger = update_elo(
        rating_challenger, rating_incumbent, score_a, elo_config.k_factor
    )

    return {
        "challenger_score": challenger_score,
        "incumbent_score": incumbent_score,
        "challenger_elo": rating_challenger,
    }


def validate(model, dataloader, device: Optional[torch.device] = None) -> Dict[str, float]:
    device = device or torch.device("cpu")
    model.eval()

    total = 0
    legal_correct = 0
    exact_correct = 0

    with torch.no_grad():
        for batch in dataloader:
            boards = batch["board"].to(device)
            moves = batch["move"].to(device)
            legal_mask = batch["legal_mask"].to(device)

            policy_logits, _ = model(boards)
            preds = torch.argmax(policy_logits, dim=-1)

            legal_hits = legal_mask.gather(1, preds.unsqueeze(1)).squeeze(1)
            legal_correct += int(legal_hits.sum().item())
            exact_correct += int((preds == moves).sum().item())
            total += boards.size(0)

    return {
        "legal_move_accuracy": legal_correct / max(1, total),
        "exact_match_accuracy": exact_correct / max(1, total),
    }


class Arena:
    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cpu")

    def play_match(
        self,
        model_a,
        model_b,
        num_games: int,
        mcts_sims: int,
        opening_temp_moves: int = 4,
    ) -> Tuple[int, int, int]:
        wins = draws = losses = 0

        model_a.eval()
        model_b.eval()

        for game_idx in range(num_games):
            if game_idx % 2 == 0:
                result = self._play_game(model_a, model_b, mcts_sims, opening_temp_moves)
                if result == 1:
                    wins += 1
                elif result == -1:
                    losses += 1
                else:
                    draws += 1
            else:
                result = self._play_game(model_b, model_a, mcts_sims, opening_temp_moves)
                if result == 1:
                    losses += 1
                elif result == -1:
                    wins += 1
                else:
                    draws += 1

        return wins, draws, losses

    def compare_models(
        self,
        challenger,
        champion,
        num_games: int,
        mcts_sims: int,
        win_rate_threshold: float = 0.55,
    ) -> Dict[str, float]:
        wins, draws, losses = self.play_match(challenger, champion, num_games, mcts_sims)
        total = max(1, wins + draws + losses)
        win_rate = wins / total
        return {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": win_rate,
            "promote": win_rate > win_rate_threshold,
        }

    def _play_game(
        self,
        white_model,
        black_model,
        num_simulations: int,
        opening_temp_moves: int,
        max_moves: int = 200,
    ) -> int:
        board = chess.Board()
        mcts_white = MCTS(white_model, config=MCTSConfig(num_simulations=num_simulations), device=self.device)
        mcts_black = MCTS(black_model, config=MCTSConfig(num_simulations=num_simulations), device=self.device)

        move_count = 0
        while not board.is_game_over() and move_count < max_moves:
            if board.turn == chess.WHITE:
                temp = 1.0 if move_count < opening_temp_moves else 0.0
                move_index = mcts_white.select_move(board, temperature=temp)
            else:
                temp = 1.0 if move_count < opening_temp_moves else 0.0
                move_index = mcts_black.select_move(board, temperature=temp)

            move = decode_move(move_index, board=board)
            board.push(move)
            move_count += 1

        result = board.result()
        if result == "1-0":
            return 1
        if result == "0-1":
            return -1
        return 0
