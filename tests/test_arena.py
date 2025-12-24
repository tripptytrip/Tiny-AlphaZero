import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from training.evaluation import Arena


class DummyModel(torch.nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        policy = torch.zeros((batch_size, 4096), device=x.device)
        value = torch.zeros((batch_size, 3), device=x.device)
        return policy, value


def test_arena_match_runs():
    arena = Arena()
    model_a = DummyModel()
    model_b = DummyModel()

    wins, draws, losses = arena.play_match(model_a, model_b, num_games=2, mcts_sims=5)

    assert wins + draws + losses == 2
