import sys
from pathlib import Path

import chess
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from mcts.tree import MCTS, MCTSConfig
from data.encoding import encode_move


class DummyModel(torch.nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        policy = torch.zeros((batch_size, 4096), device=x.device)
        value = torch.zeros((batch_size, 3), device=x.device)
        return policy, value


def test_mcts_finds_mate_in_one():
    board = chess.Board("8/8/8/8/8/5Q1K/8/6k1 w - - 0 1")
    model = DummyModel()
    mcts = MCTS(model, config=MCTSConfig(num_simulations=400, c_puct=1.5))

    visits = mcts.run(board)
    mate_move = chess.Move.from_uci("f3g2")
    mate_index = encode_move(mate_move)

    assert mate_index in visits

    assert visits[mate_index] == max(visits.values())

    board.push(mate_move)
    assert board.is_checkmate()


def test_root_dirichlet_noise():
    board = chess.Board()
    model = DummyModel()

    torch.manual_seed(0)
    config_no_noise = MCTSConfig(num_simulations=0, dirichlet_frac=0.0)
    mcts_no_noise = MCTS(model, config=config_no_noise)
    root_no_noise = mcts_no_noise.run(board)

    torch.manual_seed(0)
    config_noise = MCTSConfig(num_simulations=0, dirichlet_frac=0.25, dirichlet_alpha=0.3)
    mcts_noise = MCTS(model, config=config_noise)
    root_noise = mcts_noise.run(board)

    assert root_no_noise.keys() == root_noise.keys()
    assert root_no_noise != root_noise
