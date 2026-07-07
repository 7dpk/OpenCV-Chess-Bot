import chess
import cv2
import numpy as np

from .squares import get_square_img, row_col_to_square_name, square_to_row_col

EMPTY_STD = 15.0
CHANGE_MEAN = 10.0

_CASTLES = [
    ({"e8", "h8"}, {"f8", "g8"}, "e8g8"),
    ({"e8", "a8"}, {"c8", "d8"}, "e8c8"),
    ({"e1", "h1"}, {"f1", "g1"}, "e1g1"),
    ({"e1", "a1"}, {"c1", "d1"}, "e1c1"),
]


def is_empty(square_img: np.ndarray) -> bool:
    return float(square_img.std()) < EMPTY_STD


def board_changed(old: np.ndarray, new: np.ndarray) -> bool:
    for r in range(8):
        for c in range(8):
            if cv2.absdiff(get_square_img(r, c, old), get_square_img(r, c, new)).mean() > CHANGE_MEAN:
                return True
    return False


def _diff_squares(old, new, white_at_bottom):
    starts, ends = [], []
    for r in range(8):
        for c in range(8):
            old_sq = get_square_img(r, c, old)
            new_sq = get_square_img(r, c, new)
            if cv2.absdiff(old_sq, new_sq).mean() <= CHANGE_MEAN:
                continue
            name = row_col_to_square_name(r, c, white_at_bottom)
            if is_empty(new_sq):
                if not is_empty(old_sq):
                    starts.append(name)
            else:
                ends.append(name)
    return starts, ends


def find_candidate_moves(old, new, white_at_bottom: bool, board: chess.Board) -> list[str]:
    starts, ends = _diff_squares(old, new, white_at_bottom)
    for start_req, end_req, uci in _CASTLES:
        if start_req <= set(starts) and end_req <= set(ends):
            return [uci]
    moves = [s + e for s in starts for e in ends]
    if len(moves) > 20:
        return []
    valid = []
    for move in moves:
        if chess.Move.from_uci(move + "q") in board.legal_moves:
            return [move + "q"]
        if chess.Move.from_uci(move) in board.legal_moves:
            valid.append(move)
    if len(valid) == 2 and board.move_stack:
        last_to = chess.square_name(board.peek().to_square)
        return [valid[0]] if valid[0][2:4] != last_to else [valid[1]]
    if len(valid) > 2:
        return []
    return valid


def detect_stealth_move(img, board: chess.Board, white_at_bottom: bool) -> str | None:
    """Position drifted without a frame diff being observed: compare board state to image."""
    start, end = None, None
    for sq in chess.SQUARES:
        r, c = square_to_row_col(sq, white_at_bottom)
        img_empty = is_empty(get_square_img(r, c, img))
        if board.piece_at(sq) is not None and img_empty:
            start = chess.square_name(sq)
        if board.piece_at(sq) is None and not img_empty:
            end = chess.square_name(sq)
    if start and end:
        move = start + end
        if chess.Move.from_uci(move) in board.legal_moves:
            return move
    return None
