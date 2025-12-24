"""Train the Phase 1 model on supervised data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from training.phase1 import main as train_main  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/phase1.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--format", type=str, default=None, choices=["json", "memmap"])
    parser.add_argument("--json-path", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    argv = ["--config", args.config]
    if args.epochs is not None:
        argv += ["--epochs", str(args.epochs)]
    if args.data_dir is not None:
        argv += ["--data-dir", args.data_dir]
    if args.format is not None:
        argv += ["--format", args.format]
    if args.json_path is not None:
        argv += ["--json-path", args.json_path]
    if args.num_workers is not None:
        argv += ["--num-workers", str(args.num_workers)]

    sys.argv = [sys.argv[0]] + argv
    train_main()


if __name__ == "__main__":
    main()
