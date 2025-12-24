"""Dataset utilities for Phase 1 training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class MemmapSpec:
    boards_path: Path
    moves_path: Path
    outcomes_path: Path
    legal_moves_path: Path
    legal_offsets_path: Path
    num_positions: int


class ChessMemmapDataset(Dataset):
    """Memory-mapped dataset for large-scale training."""

    def __init__(self, spec: MemmapSpec) -> None:
        self.spec = spec
        self.boards = np.memmap(
            spec.boards_path, dtype=np.int8, mode="r", shape=(spec.num_positions, 69)
        )
        self.moves = np.memmap(
            spec.moves_path, dtype=np.int16, mode="r", shape=(spec.num_positions,)
        )
        self.outcomes = np.memmap(
            spec.outcomes_path, dtype=np.int8, mode="r", shape=(spec.num_positions,)
        )
        self.legal_moves = np.memmap(
            spec.legal_moves_path, dtype=np.int16, mode="r"
        )
        self.legal_offsets = np.memmap(
            spec.legal_offsets_path, dtype=np.int32, mode="r", shape=(spec.num_positions + 1,)
        )

    def __len__(self) -> int:
        return self.spec.num_positions

    def __getitem__(self, idx: int) -> dict:
        board = torch.from_numpy(self.boards[idx].astype(np.int64))
        move = torch.tensor(int(self.moves[idx]), dtype=torch.long)
        outcome = torch.tensor(int(self.outcomes[idx]), dtype=torch.long)

        start = int(self.legal_offsets[idx])
        end = int(self.legal_offsets[idx + 1])
        legal = torch.from_numpy(self.legal_moves[start:end].astype(np.int64))

        return {
            "board": board,
            "move": move,
            "outcome": outcome,
            "legal_moves": legal,
        }


class ChessJsonDataset(Dataset):
    """JSON dataset fallback for small-scale runs."""

    def __init__(self, json_path: Path) -> None:
        payload = json.loads(json_path.read_text())
        self.samples: Sequence[dict] = payload["samples"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        board = torch.tensor(sample["board"], dtype=torch.long)
        move = torch.tensor(sample["move"], dtype=torch.long)
        outcome = torch.tensor(sample["outcome"], dtype=torch.long)
        legal = torch.tensor(sample["legal_moves"], dtype=torch.long)
        return {
            "board": board,
            "move": move,
            "outcome": outcome,
            "legal_moves": legal,
        }


def load_memmap_spec(data_dir: Path) -> MemmapSpec:
    meta_path = data_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    num_positions = int(meta["num_positions"])

    return MemmapSpec(
        boards_path=data_dir / "boards.dat",
        moves_path=data_dir / "moves.dat",
        outcomes_path=data_dir / "outcomes.dat",
        legal_moves_path=data_dir / "legal_moves.dat",
        legal_offsets_path=data_dir / "legal_moves_offsets.dat",
        num_positions=num_positions,
    )


def build_legal_mask(legal_moves: List[torch.Tensor], num_moves: int = 4096) -> torch.Tensor:
    batch_size = len(legal_moves)
    mask = torch.zeros((batch_size, num_moves), dtype=torch.bool)
    for idx, moves in enumerate(legal_moves):
        if moves.numel() == 0:
            continue
        mask[idx, moves] = True
    return mask
