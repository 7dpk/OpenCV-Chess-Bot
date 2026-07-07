import chess
import numpy as np


def get_square_img(row: int, col: int, img: np.ndarray, trim_ratio: float = 0.15) -> np.ndarray:
    h, w = img.shape[:2]
    y0, y1 = int(row * h / 8), int((row + 1) * h / 8)
    x0, x1 = int(col * w / 8), int((col + 1) * w / 8)
    square = img[y0:y1, x0:x1]
    ty = int((y1 - y0) * trim_ratio)
    tx = int((x1 - x0) * trim_ratio)
    if ty and tx:
        return square[ty:-ty, tx:-tx]
    return square


def row_col_to_square_name(row: int, col: int, white_at_bottom: bool) -> str:
    if white_at_bottom:
        return chr(97 + col) + str(8 - row)
    return chr(97 + 7 - col) + str(row + 1)


def square_name_to_row_col(name: str, white_at_bottom: bool) -> tuple[int, int]:
    file_idx = ord(name[0]) - 97
    rank = int(name[1])
    if white_at_bottom:
        return 8 - rank, file_idx
    return rank - 1, 7 - file_idx


def square_to_row_col(square: int, white_at_bottom: bool) -> tuple[int, int]:
    rank = chess.square_rank(square)
    file_idx = chess.square_file(square)
    if white_at_bottom:
        return 7 - rank, file_idx
    return rank, 7 - file_idx


def square_center(name: str, white_at_bottom: bool, region: tuple) -> tuple[int, int]:
    left, top, right, bottom = region
    row, col = square_name_to_row_col(name, white_at_bottom)
    x = left + (col + 0.5) * (right - left) / 8
    y = top + (row + 0.5) * (bottom - top) / 8
    return int(x), int(y)
