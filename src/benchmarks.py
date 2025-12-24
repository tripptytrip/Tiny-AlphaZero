"""Structural benchmarks for TinyAlphaZero."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import chess
import torch

from data.encoding import decode_move, encode_board, encode_move


@dataclass
class BenchmarkCase:
    name: str
    fen: str
    legal_move_mass_min: float = 0.0
    move_prob_high: Optional[Dict[str, float]] = None
    move_prob_low: Optional[Dict[str, float]] = None
    value_sign: Optional[str] = None  # "positive", "negative", "neutral"


class ChessBenchmarkRunner:
    def __init__(self, model, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.device = device or torch.device("cpu")

    def run_single_test(self, test_case: BenchmarkCase) -> Dict[str, object]:
        board = chess.Board(test_case.fen)
        board_tensor = torch.tensor(encode_board(board), dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            policy_probs, value = self.model.predict(board_tensor)

        policy_probs = policy_probs.squeeze(0)
        legal_indices = [encode_move(m) for m in board.legal_moves]
        legal_mass = float(policy_probs[legal_indices].sum().item()) if legal_indices else 0.0

        checks: List[Dict[str, object]] = []

        if legal_mass >= test_case.legal_move_mass_min:
            checks.append(
                {
                    "detail": f"legal_move_mass >= {test_case.legal_move_mass_min:.3f}",
                    "passed": True,
                }
            )
        else:
            checks.append(
                {
                    "detail": f"legal_move_mass >= {test_case.legal_move_mass_min:.3f}",
                    "passed": False,
                }
            )

        if test_case.move_prob_high:
            for uci, min_prob in test_case.move_prob_high.items():
                move = chess.Move.from_uci(uci)
                idx = encode_move(move)
                checks.append(
                    {
                        "detail": f"{uci} prob >= {min_prob:.3f}",
                        "passed": float(policy_probs[idx]) >= min_prob,
                    }
                )

        if test_case.move_prob_low:
            for uci, max_prob in test_case.move_prob_low.items():
                move = chess.Move.from_uci(uci)
                idx = encode_move(move)
                checks.append(
                    {
                        "detail": f"{uci} prob <= {max_prob:.3f}",
                        "passed": float(policy_probs[idx]) <= max_prob,
                    }
                )

        if test_case.value_sign:
            value_scalar = float(value.item())
            if test_case.value_sign == "positive":
                checks.append({"detail": "value_sign positive", "passed": value_scalar > 0.0})
            elif test_case.value_sign == "negative":
                checks.append({"detail": "value_sign negative", "passed": value_scalar < 0.0})
            else:
                checks.append({"detail": "value_sign neutral", "passed": abs(value_scalar) < 0.1})

        passed = all(check["passed"] for check in checks)

        return {
            "name": test_case.name,
            "fen": test_case.fen,
            "passed": passed,
            "legal_move_mass": legal_mass,
            "checks": checks,
        }

    def run(self, cases: List[BenchmarkCase]) -> List[Dict[str, object]]:
        return [self.run_single_test(case) for case in cases]

    def run_all(self) -> Dict[str, object]:
        categories: Dict[str, Dict[str, object]] = {}
        passed_total = 0
        failed_total = 0

        for key, category in BENCHMARKS.items():
            results = self.run(category["tests"])
            categories[key] = {
                "name": category["name"],
                "tests": results,
            }
            for test in results:
                if test["passed"]:
                    passed_total += 1
                else:
                    failed_total += 1

        return {
            "summary": {"passed": passed_total, "failed": failed_total},
            "categories": categories,
        }


BENCHMARKS = {
    "basic_legality": {
        "name": "Basic Legality",
        "tests": [
            BenchmarkCase(
                name="Start Position",
                fen=chess.STARTING_FEN,
                legal_move_mass_min=0.0,
                move_prob_high={"e2e4": 0.0},
                move_prob_low={"e2e5": 1.0},
            ),
            BenchmarkCase(
                name="Knight Corner",
                fen="8/8/8/8/8/8/6N1/7K w - - 0 1",
                legal_move_mass_min=0.0,
                move_prob_high={"g2h4": 0.0},
            ),
        ],
    }
}


def default_benchmarks() -> List[BenchmarkCase]:
    return BENCHMARKS["basic_legality"]["tests"]
