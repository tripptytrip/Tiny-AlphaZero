import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from data.encoding import (
    BOARD_TOKENS,
    CASTLE_BK_Y,
    CASTLE_BQ_Y,
    CASTLE_WK_Y,
    CASTLE_WQ_Y,
    TURN_W,
    WK,
    WR,
    BK,
    decode_board,
    decode_move,
    encode_board,
    encode_move,
    get_legal_move_mask,
)


def test_board_encoding_roundtrip():
    board = chess.Board()
    for move in [
        chess.Move.from_uci("g1f3"),
        chess.Move.from_uci("g8f6"),
        chess.Move.from_uci("b1c3"),
        chess.Move.from_uci("b8c6"),
    ]:
        board.push(move)

    tokens = encode_board(board)
    decoded = decode_board(tokens)
    assert encode_board(decoded) == tokens


def test_move_encoding_all_legal():
    board = chess.Board()
    for move in board.legal_moves:
        idx = encode_move(move)
        assert 0 <= idx < 4096
        decoded = decode_move(idx, board=board)
        assert decoded in board.legal_moves


def test_starting_position():
    board = chess.Board()
    tokens = encode_board(board)
    assert len(tokens) == BOARD_TOKENS

    assert tokens[0] == WR  # a1
    assert tokens[4] == WK  # e1
    assert tokens[60] == BK  # e8
    assert tokens[64] == TURN_W
    assert tokens[65] == CASTLE_WK_Y
    assert tokens[66] == CASTLE_WQ_Y
    assert tokens[67] == CASTLE_BK_Y
    assert tokens[68] == CASTLE_BQ_Y


def test_legal_move_mask_matches_list():
    board = chess.Board()
    mask = get_legal_move_mask(board)
    legal_indices = {encode_move(m) for m in board.legal_moves}
    assert mask.sum() == len(legal_indices)
    for idx in legal_indices:
        assert mask[idx]
