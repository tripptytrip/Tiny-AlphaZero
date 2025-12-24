"""Generate a single self-play game with MCTS policy targets."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from data.generation import SelfPlayConfig, generate_self_play_game  # noqa: E402
from model.transformer import ChessTransformer  # noqa: E402


def get_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "mps":
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output", type=str, default="data/self_play_game.json")
    parser.add_argument("--num-sims", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    device = get_device(args.device)
    model = ChessTransformer().to(device)

    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(payload["model_state_dict"])

    config = SelfPlayConfig(num_simulations=args.num_sims)

    print(f"Running self-play on {device} with {config.mcts.num_simulations} sims/move")

    try:
        samples = generate_self_play_game(model, config=config, device=device, progress=args.progress)
    except RuntimeError as exc:
        if args.device == "auto":
            print(f"Device error on {device}, retrying on CPU: {exc}")
            device = torch.device("cpu")
            model = ChessTransformer().to(device)
            if args.checkpoint:
                payload = torch.load(args.checkpoint, map_location=device)
                model.load_state_dict(payload["model_state_dict"])
            samples = generate_self_play_game(model, config=config, device=device, progress=args.progress)
        else:
            raise

    payload = {
        "metadata": {
            "version": "1.0",
            "generated": datetime.now(timezone.utc).isoformat(),
            "num_positions": len(samples),
        },
        "samples": samples,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload))

    print(f"Wrote self-play game with {len(samples)} positions to {output_path}")


if __name__ == "__main__":
    main()
