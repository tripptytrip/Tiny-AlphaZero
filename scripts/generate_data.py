"""Generate random self-play games for Phase 1."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from data.generation import GenerationConfig, generate_games


def build_metadata(num_games: int, num_positions: int) -> dict:
    return {
        "version": "1.0",
        "generated": datetime.now(timezone.utc).isoformat(),
        "num_games": num_games,
        "num_positions": num_positions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--max-game-length", type=int, default=200)
    parser.add_argument("--min-game-length", type=int, default=10)
    parser.add_argument("--biased", action="store_true")
    parser.add_argument("--capture-weight", type=float, default=3.0)
    parser.add_argument("--check-weight", type=float, default=2.0)
    parser.add_argument("--output", type=str, default="data/sample_games.json")
    args = parser.parse_args()

    config = GenerationConfig(
        max_game_length=args.max_game_length,
        min_game_length=args.min_game_length,
        biased_sampling=args.biased,
        capture_weight=args.capture_weight,
        check_weight=args.check_weight,
    )

    games = generate_games(args.num_games, config=config)
    samples = [pos for game in games for pos in game]

    payload = {
        "metadata": build_metadata(args.num_games, len(samples)),
        "samples": samples,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload))

    print(f"Wrote {args.num_games} games with {len(samples)} positions to {output_path}")


if __name__ == "__main__":
    main()
