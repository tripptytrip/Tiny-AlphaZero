"""Run Phase 2 self-play training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from training.phase2 import main as train_main  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-games", type=int, default=2)
    parser.add_argument("--mcts-sims", type=int, default=10)
    parser.add_argument("--training-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-game-length", type=int, default=200)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--use-batched", action="store_true")
    parser.add_argument("--batched-games", type=int, default=4)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-frac", type=float, default=0.25)
    parser.add_argument("--random-opening-ply", type=int, default=10)
    parser.add_argument("--random-opening-prob", type=float, default=0.5)
    parser.add_argument("--self-play-ratio", type=float, default=0.8)
    parser.add_argument("--pool-size", type=int, default=20)
    args = parser.parse_args()

    argv = ["--device", args.device]
    if args.checkpoint:
        argv += ["--checkpoint", args.checkpoint]
    argv += [
        "--num-games",
        str(args.num_games),
        "--mcts-sims",
        str(args.mcts_sims),
        "--training-steps",
        str(args.training_steps),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--max-game-length",
        str(args.max_game_length),
        "--log-every",
        str(args.log_every),
        "--batched-games",
        str(args.batched_games),
        "--dirichlet-alpha",
        str(args.dirichlet_alpha),
        "--dirichlet-frac",
        str(args.dirichlet_frac),
        "--random-opening-ply",
        str(args.random_opening_ply),
        "--random-opening-prob",
        str(args.random_opening_prob),
        "--self-play-ratio",
        str(args.self_play_ratio),
        "--pool-size",
        str(args.pool_size),
    ]
    if args.progress:
        argv.append("--progress")
    if args.use_batched:
        argv.append("--use-batched")

    sys.argv = [sys.argv[0]] + argv
    train_main()


if __name__ == "__main__":
    main()
