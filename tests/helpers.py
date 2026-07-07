import cv2
import numpy as np

from chessbot.vision.squares import square_name_to_row_col

LIGHT, DARK, PIECE = 200, 110, 30


def render_fake_board(occupied: set[str], white_at_bottom: bool = True, size: int = 512) -> np.ndarray:
    img = np.zeros((size, size), np.uint8)
    sq = size // 8
    for r in range(8):
        for c in range(8):
            img[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = LIGHT if (r + c) % 2 == 0 else DARK
    for name in occupied:
        r, c = square_name_to_row_col(name, white_at_bottom)
        center = (int((c + 0.5) * sq), int((r + 0.5) * sq))
        cv2.circle(img, center, int(sq * 0.3), PIECE, -1)
    return img


def occupied_squares(board) -> set[str]:
    import chess

    return {chess.square_name(sq) for sq in board.piece_map()}
