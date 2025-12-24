import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from model.encoding import ChessPositionEncoding
from model.transformer import ChessTransformer


def test_position_encoding_file_component_shared():
    encoder = ChessPositionEncoding(d_model=8)
    positions = torch.tensor([0, 8])
    embeddings = encoder(positions)

    ranks = torch.tensor([0, 1])
    files = torch.tensor([0, 0])
    expected = encoder.rank_embed(ranks) + encoder.file_embed(files)

    assert torch.allclose(embeddings, expected)


def test_model_forward_shapes_and_value_range():
    model = ChessTransformer()
    batch = torch.randint(0, 23, (4, 69))

    policy, value_logits = model(batch)
    values = model.get_value(value_logits)

    assert policy.shape == (4, 4096)
    assert value_logits.shape == (4, 3)
    assert torch.all(values <= 1.0)
    assert torch.all(values >= -1.0)
