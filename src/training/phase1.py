"""Phase 1 supervised training loop."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from data.dataset import (  # noqa: E402
    ChessJsonDataset,
    ChessMemmapDataset,
    build_legal_mask,
    load_memmap_spec,
)
from model.transformer import ChessTransformer  # noqa: E402
from training.evaluation import validate  # noqa: E402


@dataclass
class Phase1Trainer:
    config: dict
    model: ChessTransformer
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None

    def compute_losses(
        self,
        policy_logits: torch.Tensor,
        value_logits: torch.Tensor,
        moves: torch.Tensor,
        outcomes: torch.Tensor,
        policy_weight: float,
        value_weight: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        policy_loss = nn.functional.cross_entropy(policy_logits, moves)
        outcome_class = outcomes + 1
        value_loss = nn.functional.cross_entropy(value_logits, outcome_class)
        total = policy_weight * policy_loss + value_weight * value_loss
        return policy_loss, value_loss, total

    def train_epoch(self, loader: DataLoader, device: torch.device) -> Dict[str, float]:
        self.model.train()
        policy_weight = float(self.config["training"].get("policy_weight", 1.0))
        value_weight = float(self.config["training"].get("value_weight", 1.0))
        grad_clip = float(self.config["training"].get("grad_clip", 1.0))

        total_policy = 0.0
        total_value = 0.0
        total_loss = 0.0
        total_count = 0

        log_every = int(self.config.get("logging", {}).get("log_every", 100))
        for batch_idx, batch in enumerate(loader, start=1):
            boards = batch["board"].to(device)
            moves = batch["move"].to(device)
            outcomes = batch["outcome"].to(device)

            self.optimizer.zero_grad(set_to_none=True)
            policy_logits, value_logits = self.model(boards)
            loss_policy, loss_value, loss_total = self.compute_losses(
                policy_logits, value_logits, moves, outcomes, policy_weight, value_weight
            )
            loss_total.backward()
            policy_grad = self.model.policy_head[0].weight.grad
            value_grad = self.model.value_head[0].weight.grad
            policy_grad_norm = float(policy_grad.norm().item()) if policy_grad is not None else 0.0
            value_grad_norm = float(value_grad.norm().item()) if value_grad is not None else 0.0
            if batch_idx % log_every == 0:
                print(
                    f"Grad norms | policy {policy_grad_norm:.4f} | value {value_grad_norm:.4f}"
                )
            nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            batch_size = boards.size(0)
            total_policy += loss_policy.item() * batch_size
            total_value += loss_value.item() * batch_size
            total_loss += loss_total.item() * batch_size
            total_count += batch_size

        return {
            "loss_policy": total_policy / total_count,
            "loss_value": total_value / total_count,
            "total_loss": total_loss / total_count,
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> None:
        epochs = int(self.config["training"].get("epochs", 1))
        checkpoint_dir = Path(self.config["training"].get("checkpoint_dir", "checkpoints/phase1"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_loss = math.inf

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, device)
            val_metrics = validate(self.model, val_loader, device=device)

            print(
                f"Epoch {epoch:03d} | loss {train_metrics['total_loss']:.4f} | "
                f"policy {train_metrics['loss_policy']:.4f} | value {train_metrics['loss_value']:.4f} | "
                f"legal {val_metrics['legal_move_accuracy']:.4f} | exact {val_metrics['exact_match_accuracy']:.4f}"
            )

            latest_path = checkpoint_dir / "latest.pt"
            torch.save({"model_state_dict": self.model.state_dict()}, latest_path)

            if train_metrics["total_loss"] < best_loss:
                best_loss = train_metrics["total_loss"]
                best_path = checkpoint_dir / "best.pt"
                torch.save({"model_state_dict": self.model.state_dict()}, best_path)


def load_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text())


def build_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    batch_size = config["training"]["batch_size"]
    num_workers = data_cfg.get("num_workers", 4)
    val_split = float(data_cfg.get("val_split", 0.1))
    seed = int(data_cfg.get("seed", 42))

    if data_cfg.get("format", "memmap") == "json":
        dataset = ChessJsonDataset(Path(data_cfg["json_path"]))
    else:
        spec = load_memmap_spec(Path(data_cfg["data_dir"]))
        dataset = ChessMemmapDataset(spec)

    total = len(dataset)
    val_size = max(1, int(total * val_split))
    indices = torch.randperm(total, generator=torch.Generator().manual_seed(seed))
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    return train_loader, val_loader


def collate_batch(batch: list[dict]) -> Dict[str, torch.Tensor]:
    boards = torch.stack([item["board"] for item in batch])
    moves = torch.stack([item["move"] for item in batch])
    outcomes = torch.stack([item["outcome"] for item in batch])
    legal_moves = [item["legal_moves"] for item in batch]
    legal_mask = build_legal_mask(legal_moves)

    return {
        "board": boards,
        "move": moves,
        "outcome": outcomes,
        "legal_mask": legal_mask,
    }


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/phase1.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--format", type=str, default=None, choices=["json", "memmap"])
    parser.add_argument("--json-path", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.data_dir is not None:
        config["data"]["data_dir"] = args.data_dir
    if args.format is not None:
        config["data"]["format"] = args.format
    if args.json_path is not None:
        config["data"]["json_path"] = args.json_path
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers

    device = get_device()
    model_cfg = config["model"]
    model = ChessTransformer(
        vocab_size=model_cfg.get("vocab_size", 23),
        num_moves=model_cfg.get("num_moves", 4096),
        d_model=model_cfg.get("d_model", 256),
        n_layers=model_cfg.get("n_layers", 6),
        n_heads=model_cfg.get("n_heads", 8),
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)

    train_loader, val_loader = build_dataloaders(config)

    training_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )

    trainer = Phase1Trainer(config=config, model=model, optimizer=optimizer)
    trainer.fit(train_loader, val_loader, device)


if __name__ == "__main__":
    main()
