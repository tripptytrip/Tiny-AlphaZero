"""Convert JSON dataset into memmap files for Phase 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def convert(json_path: Path, output_dir: Path) -> None:
    payload = json.loads(json_path.read_text())
    samples = payload["samples"]
    num_positions = len(samples)

    output_dir.mkdir(parents=True, exist_ok=True)

    boards_path = output_dir / "boards.dat"
    moves_path = output_dir / "moves.dat"
    outcomes_path = output_dir / "outcomes.dat"
    legal_moves_path = output_dir / "legal_moves.dat"
    legal_offsets_path = output_dir / "legal_moves_offsets.dat"

    legal_lengths = [len(sample["legal_moves"]) for sample in samples]
    total_legal = sum(legal_lengths)

    boards = np.memmap(boards_path, dtype=np.int8, mode="w+", shape=(num_positions, 69))
    moves = np.memmap(moves_path, dtype=np.int16, mode="w+", shape=(num_positions,))
    outcomes = np.memmap(outcomes_path, dtype=np.int8, mode="w+", shape=(num_positions,))
    legal_moves = np.memmap(legal_moves_path, dtype=np.int16, mode="w+", shape=(total_legal,))
    legal_offsets = np.memmap(
        legal_offsets_path, dtype=np.int32, mode="w+", shape=(num_positions + 1,)
    )

    offset = 0
    for idx, sample in enumerate(samples):
        boards[idx] = np.array(sample["board"], dtype=np.int8)
        moves[idx] = int(sample["move"])
        outcomes[idx] = int(sample["outcome"])

        legal_offsets[idx] = offset
        legal = np.array(sample["legal_moves"], dtype=np.int16)
        legal_moves[offset : offset + len(legal)] = legal
        offset += len(legal)

    legal_offsets[num_positions] = offset

    boards.flush()
    moves.flush()
    outcomes.flush()
    legal_moves.flush()
    legal_offsets.flush()

    meta = {
        "version": payload.get("metadata", {}).get("version", "1.0"),
        "num_positions": num_positions,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta))

    print(f"Wrote memmap dataset with {num_positions} positions to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/sample_games.json")
    parser.add_argument("--output-dir", type=str, default="data/phase1")
    args = parser.parse_args()

    convert(Path(args.input), Path(args.output_dir))


if __name__ == "__main__":
    main()
