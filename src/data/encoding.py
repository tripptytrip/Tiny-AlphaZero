"""Board and move encoding utilities for TinyAlphaZero."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import chess
import numpy as np

# Token IDs
ES = 0
WP = 1
WN = 2
WB = 3
WR = 4
WQ = 5
WK = 6
BP = 7
BN = 8
BB = 9
BR = 10
BQ = 11
BK = 12

TURN_W = 13
TURN_B = 14
CASTLE_WK_Y = 15
CASTLE_WK_N = 16
CASTLE_WQ_Y = 17
CASTLE_WQ_N = 18
CASTLE_BK_Y = 19
CASTLE_BK_N = 20
CASTLE_BQ_Y = 21
CASTLE_BQ_N = 22

VOCAB_SIZE = 23
BOARD_TOKENS = 69
MOVE_SPACE_SIZE = 64 * 64

PIECE_TO_TOKEN = {
    None: ES,
    (chess.PAWN, chess.WHITE): WP,
    (chess.KNIGHT, chess.WHITE): WN,
    (chess.BISHOP, chess.WHITE): WB,
    (chess.ROOK, chess.WHITE): WR,
    (chess.QUEEN, chess.WHITE): WQ,
    (chess.KING, chess.WHITE): WK,
    (chess.PAWN, chess.BLACK): BP,
    (chess.KNIGHT, chess.BLACK): BN,
    (chess.BISHOP, chess.BLACK): BB,
    (chess.ROOK, chess.BLACK): BR,
    (chess.QUEEN, chess.BLACK): BQ,
    (chess.KING, chess.BLACK): BK,
}

TOKEN_TO_PIECE = {
    ES: None,
    WP: chess.Piece(chess.PAWN, chess.WHITE),
    WN: chess.Piece(chess.KNIGHT, chess.WHITE),
    WB: chess.Piece(chess.BISHOP, chess.WHITE),
    WR: chess.Piece(chess.ROOK, chess.WHITE),
    WQ: chess.Piece(chess.QUEEN, chess.WHITE),
    WK: chess.Piece(chess.KING, chess.WHITE),
    BP: chess.Piece(chess.PAWN, chess.BLACK),
    BN: chess.Piece(chess.KNIGHT, chess.BLACK),
    BB: chess.Piece(chess.BISHOP, chess.BLACK),
    BR: chess.Piece(chess.ROOK, chess.BLACK),
    BQ: chess.Piece(chess.QUEEN, chess.BLACK),
    BK: chess.Piece(chess.KING, chess.BLACK),
}


@dataclass(frozen=True)
class EncodedPosition:
    board_tokens: List[int]
    move_index: Optional[int] = None
    legal_move_indices: Optional[List[int]] = None
    outcome: Optional[int] = None


def encode_board(board: chess.Board) -> List[int]:
    """Encode a chess.Board into a fixed-length list of token IDs."""
    tokens: List[int] = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            tokens.append(ES)
        else:
            tokens.append(PIECE_TO_TOKEN[(piece.piece_type, piece.color)])

    tokens.append(TURN_W if board.turn == chess.WHITE else TURN_B)

    tokens.append(CASTLE_WK_Y if board.has_kingside_castling_rights(chess.WHITE) else CASTLE_WK_N)
    tokens.append(CASTLE_WQ_Y if board.has_queenside_castling_rights(chess.WHITE) else CASTLE_WQ_N)
    tokens.append(CASTLE_BK_Y if board.has_kingside_castling_rights(chess.BLACK) else CASTLE_BK_N)
    tokens.append(CASTLE_BQ_Y if board.has_queenside_castling_rights(chess.BLACK) else CASTLE_BQ_N)

    return tokens


def decode_board(tokens: Iterable[int]) -> chess.Board:
    """Decode a token list into a chess.Board. En passant is not preserved."""
    tokens = list(tokens)
    if len(tokens) != BOARD_TOKENS:
        raise ValueError(f"Expected {BOARD_TOKENS} tokens, got {len(tokens)}")

    board = chess.Board(None)

    for square, token in zip(chess.SQUARES, tokens[:64]):
        piece = TOKEN_TO_PIECE.get(token)
        if piece is not None:
            board.set_piece_at(square, piece)

    board.turn = tokens[64] == TURN_W

    rights = 0
    if tokens[65] == CASTLE_WK_Y:
        rights |= chess.BB_H1
    if tokens[66] == CASTLE_WQ_Y:
        rights |= chess.BB_A1
    if tokens[67] == CASTLE_BK_Y:
        rights |= chess.BB_H8
    if tokens[68] == CASTLE_BQ_Y:
        rights |= chess.BB_A8
    board.castling_rights = rights

    board.ep_square = None
    board.clear_stack()
    return board


def encode_move(move: chess.Move) -> int:
    """Encode a move as from_square * 64 + to_square."""
    return move.from_square * 64 + move.to_square


def decode_move(move_index: int, board: Optional[chess.Board] = None) -> chess.Move:
    """Decode a move index into a chess.Move. Promotes to queen if needed."""
    if move_index < 0 or move_index >= MOVE_SPACE_SIZE:
        raise ValueError(f"Move index out of range: {move_index}")

    from_sq = move_index // 64
    to_sq = move_index % 64

    if board is None:
        return chess.Move(from_sq, to_sq)

    piece = board.piece_at(from_sq)
    if piece is not None and piece.piece_type == chess.PAWN:
        rank = chess.square_rank(to_sq)
        if rank == 0 or rank == 7:
            return chess.Move(from_sq, to_sq, promotion=chess.QUEEN)

    return chess.Move(from_sq, to_sq)


def get_legal_move_indices(board: chess.Board) -> List[int]:
    """Return encoded indices for all legal moves in the position."""
    return [encode_move(move) for move in board.legal_moves]


def get_legal_move_mask(board: chess.Board) -> np.ndarray:
    """Return a boolean mask over the 4096 move space for legal moves."""
    mask = np.zeros(MOVE_SPACE_SIZE, dtype=bool)
    for move in board.legal_moves:
        mask[encode_move(move)] = True
    return mask
