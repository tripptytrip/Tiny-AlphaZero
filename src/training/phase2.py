"""Phase 2 self-play training loop."""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as mp
import time
import os
import sys
import tempfile
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from data.generation import SelfPlayConfig, generate_self_play_game  # noqa: E402
from mcts.batched import generate_self_play_batch  # noqa: E402
from model.transformer import ChessTransformer  # noqa: E402
from training.evaluation import Arena  # noqa: E402
from training.pool import CheckpointPool, CheckpointPoolConfig  # noqa: E402
from training.replay import ReplayBuffer, ReplayBufferConfig  # noqa: E402


@dataclass
class Phase2Config:
    num_games: int = 2
    mcts_sims: int = 10
    training_steps: int = 5
    batch_size: int = 64
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    value_weight: float = 1.0
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints/phase2"
    max_workers: int = 2
    arena_games: int = 4
    arena_sims: int = 50
    win_rate_threshold: float = 0.55
    max_game_length: int = 200
    progress: bool = False
    log_every: int = 5
    use_batched: bool = False
    batched_games: int = 4
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    random_opening_ply: int = 10
    random_opening_prob: float = 0.5
    self_play_ratio: float = 0.8
    pool_size: int = 20


def self_play_worker(
    num_simulations: int,
    max_game_length: int,
    dirichlet_alpha: float,
    dirichlet_frac: float,
    random_opening_ply: int,
    random_opening_prob: float,
    checkpoint_path: str,
    device: str,
) -> List[dict]:
    torch_device = torch.device(device)
    model = ChessTransformer().to(torch_device)
    payload = torch.load(checkpoint_path, map_location=torch_device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    config = SelfPlayConfig(
        num_simulations=num_simulations,
        max_game_length=max_game_length,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_frac=dirichlet_frac,
        random_opening_ply=random_opening_ply,
        random_opening_prob=random_opening_prob,
    )
    return generate_self_play_game(model, config=config, device=torch_device)


def get_device(requested: str = "auto") -> torch.device:
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


def policy_loss(policy_logits: torch.Tensor, target_policy: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(policy_logits, dim=-1)
    return -(target_policy * log_probs).sum(dim=-1).mean()


def value_loss(value_logits: torch.Tensor, outcomes: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(value_logits, outcomes)


class Phase2Trainer:
    def __init__(
        self,
        model: ChessTransformer,
        config: Phase2Config,
        device: torch.device,
    ) -> None:
        self.current_model = model
        self.best_model = ChessTransformer().to(device)
        self.best_model.load_state_dict(model.state_dict())
        self.best_model.eval()
        self.device = device
        self.config = config

        self.replay = ReplayBuffer(ReplayBufferConfig())
        self.checkpoint_pool = CheckpointPool(CheckpointPoolConfig(max_size=config.pool_size))
        self.optimizer = torch.optim.AdamW(
            self.current_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def _select_self_play_model(self) -> ChessTransformer:
        use_best = True
        if len(self.checkpoint_pool) > 0:
            use_best = random.random() < self.config.self_play_ratio

        if use_best:
            return self.best_model

        checkpoint_path = self.checkpoint_pool.sample()
        if checkpoint_path is None:
            return self.best_model

        opponent = ChessTransformer().to(self.device)
        payload = torch.load(checkpoint_path, map_location=self.device)
        opponent.load_state_dict(payload["model_state_dict"])
        opponent.eval()
        return opponent

    def _generate_games(self) -> List[List[dict]]:
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_for_self_play = self._select_self_play_model()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt", dir=checkpoint_dir) as temp_file:
            torch.save({"model_state_dict": model_for_self_play.state_dict()}, temp_file.name)
            temp_path = temp_file.name

        games: List[List[dict]] = []
        print(f"Generating {self.config.num_games} self-play games...", flush=True)
        start = time.perf_counter()
        try:
            if self.config.use_batched:
                batch_samples = generate_self_play_batch(
                    model_for_self_play,
                    batch_size=self.config.batched_games,
                    num_simulations=self.config.mcts_sims,
                    max_game_length=self.config.max_game_length,
                    device=self.device,
                    dirichlet_alpha=self.config.dirichlet_alpha,
                    dirichlet_frac=self.config.dirichlet_frac,
                    random_opening_ply=self.config.random_opening_ply,
                    random_opening_prob=self.config.random_opening_prob,
                )
                for game in batch_samples:
                    games.append(game)
                duration = max(1e-8, time.perf_counter() - start)
                total_positions = sum(len(game) for game in games)
                print(
                    f"Self-play throughput: {total_positions / duration:.2f} positions/sec",
                    flush=True,
                )
                return games

            try:
                ctx = mp.get_context("spawn")
                try:
                    ctx.Semaphore(1)
                    use_pool = True
                except PermissionError:
                    use_pool = False

                if not use_pool or self.config.max_workers <= 1:
                    raise PermissionError("Multiprocessing semaphore unavailable")

                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.config.max_workers,
                    mp_context=ctx,
                ) as executor:
                    futures = [
                        executor.submit(
                            self_play_worker,
                            self.config.mcts_sims,
                            self.config.max_game_length,
                            self.config.dirichlet_alpha,
                            self.config.dirichlet_frac,
                            self.config.random_opening_ply,
                            self.config.random_opening_prob,
                            temp_path,
                            str(self.device),
                        )
                        for _ in range(self.config.num_games)
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        games.append(future.result())
                duration = max(1e-8, time.perf_counter() - start)
                total_positions = sum(len(game) for game in games)
                print(
                    f"Self-play throughput: {total_positions / duration:.2f} positions/sec",
                    flush=True,
                )
            except PermissionError as exc:
                print(
                    f"Process pool unavailable, trying thread pool: {exc}",
                    flush=True,
                )
                try:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=self.config.max_workers
                    ) as executor:
                        futures = [
                            executor.submit(
                                self_play_worker,
                                self.config.mcts_sims,
                                self.config.max_game_length,
                                self.config.dirichlet_alpha,
                                self.config.dirichlet_frac,
                                self.config.random_opening_ply,
                                self.config.random_opening_prob,
                                temp_path,
                                str(self.device),
                            )
                            for _ in range(self.config.num_games)
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            games.append(future.result())
                    duration = max(1e-8, time.perf_counter() - start)
                    total_positions = sum(len(game) for game in games)
                    print(
                        f"Self-play throughput: {total_positions / duration:.2f} positions/sec",
                        flush=True,
                    )
                except Exception as thread_exc:
                    print(
                        f"Thread pool unavailable, falling back to sequential self-play: {thread_exc}",
                        flush=True,
                    )
                    config = SelfPlayConfig(
                        num_simulations=self.config.mcts_sims,
                        max_game_length=self.config.max_game_length,
                        dirichlet_alpha=self.config.dirichlet_alpha,
                        dirichlet_frac=self.config.dirichlet_frac,
                        random_opening_ply=self.config.random_opening_ply,
                        random_opening_prob=self.config.random_opening_prob,
                    )
                    for idx in range(self.config.num_games):
                        print(
                            f"Sequential self-play game {idx + 1}/{self.config.num_games}...",
                            flush=True,
                        )
                        games.append(
                            generate_self_play_game(
                                model_for_self_play,
                                config=config,
                                device=self.device,
                                progress=self.config.progress,
                                log_every=self.config.log_every,
                            )
                        )
                    duration = max(1e-8, time.perf_counter() - start)
                    total_positions = sum(len(game) for game in games)
                    print(
                        f"Self-play throughput: {total_positions / duration:.2f} positions/sec",
                        flush=True,
                    )
        finally:
            os.unlink(temp_path)

        return games

    def train_iteration(self, iteration: int = 0) -> None:
        print(f"Starting iteration {iteration}", flush=True)
        games = self._generate_games()
        for game in games:
            self.replay.push(game)

        total_positions = sum(len(game) for game in games)
        print(f"Generated {total_positions} positions", flush=True)

        if len(self.replay) < self.config.batch_size:
            print("Replay buffer too small for training batch, skipping optimization")
            return

        self.current_model.train()
        for step in range(self.config.training_steps):
            boards, policies, outcomes = self.replay.sample(self.config.batch_size)
            boards = boards.to(self.device)
            policies = policies.to(self.device)
            outcomes = outcomes.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            policy_logits, value_logits = self.current_model(boards)
            loss_policy = policy_loss(policy_logits, policies)
            loss_value = value_loss(value_logits, outcomes)
            loss = loss_policy + self.config.value_weight * loss_value
            loss.backward()
            policy_grad = self.current_model.policy_head[0].weight.grad
            value_grad = self.current_model.value_head[0].weight.grad
            policy_grad_norm = float(policy_grad.norm().item()) if policy_grad is not None else 0.0
            value_grad_norm = float(value_grad.norm().item()) if value_grad is not None else 0.0
            torch.nn.utils.clip_grad_norm_(self.current_model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            if step == 0 or (step + 1) % 5 == 0:
                print(
                    f"Iter {iteration} Step {step + 1}/{self.config.training_steps} | "
                    f"policy {loss_policy.item():.4f} | value {loss_value.item():.4f} | "
                    f"grad_p {policy_grad_norm:.4f} | grad_v {value_grad_norm:.4f}"
                )

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f"phase2_iter_{iteration}.pt"
        torch.save({"model_state_dict": self.current_model.state_dict()}, ckpt_path)
        self.checkpoint_pool.add(str(ckpt_path))
        print(f"Saved checkpoint to {ckpt_path}")

        arena = Arena(device=self.device)
        result = arena.compare_models(
            self.current_model,
            self.best_model,
            num_games=self.config.arena_games,
            mcts_sims=self.config.arena_sims,
            win_rate_threshold=self.config.win_rate_threshold,
        )

        if result["promote"]:
            self.best_model.load_state_dict(self.current_model.state_dict())
            best_path = checkpoint_dir / f"best_generation_{iteration}.pt"
            torch.save({"model_state_dict": self.best_model.state_dict()}, best_path)
            print(f"🚀 New Champion! Win Rate: {result['win_rate']:.2%}")
        else:
            print(f"Champion holds. Win Rate: {result['win_rate']:.2%}")


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

    device = get_device(args.device)
    model = ChessTransformer().to(device)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(payload["model_state_dict"])

    config = Phase2Config(
        num_games=args.num_games,
        mcts_sims=args.mcts_sims,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        max_workers=args.num_workers,
        max_game_length=args.max_game_length,
        progress=args.progress,
        log_every=args.log_every,
        use_batched=args.use_batched,
        batched_games=args.batched_games,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_frac=args.dirichlet_frac,
        random_opening_ply=args.random_opening_ply,
        random_opening_prob=args.random_opening_prob,
        self_play_ratio=args.self_play_ratio,
        pool_size=args.pool_size,
    )

    trainer = Phase2Trainer(model, config, device)
    trainer.train_iteration(iteration=0)


if __name__ == "__main__":
    main()
