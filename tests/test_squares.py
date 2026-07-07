import numpy as np
import chess

from chessbot.vision.squares import (
    get_square_img,
    row_col_to_square_name,
    square_center,
    square_name_to_row_col,
    square_to_row_col,
)


def test_row_col_to_square_name_white_bottom():
    assert row_col_to_square_name(7, 0, True) == "a1"
    assert row_col_to_square_name(0, 0, True) == "a8"
    assert row_col_to_square_name(6, 4, True) == "e2"


def test_row_col_to_square_name_black_bottom():
    assert row_col_to_square_name(0, 0, False) == "h1"
    assert row_col_to_square_name(7, 7, False) == "a8"
    assert row_col_to_square_name(1, 3, False) == "e2"


def test_square_name_roundtrip():
    for white_at_bottom in (True, False):
        for row in range(8):
            for col in range(8):
                name = row_col_to_square_name(row, col, white_at_bottom)
                assert square_name_to_row_col(name, white_at_bottom) == (row, col)


def test_square_to_row_col_matches_name_mapping():
    for white_at_bottom in (True, False):
        for sq in chess.SQUARES:
            row, col = square_to_row_col(sq, white_at_bottom)
            assert row_col_to_square_name(row, col, white_at_bottom) == chess.square_name(sq)


def test_get_square_img_shape_and_trim():
    img = np.arange(160 * 160, dtype=np.uint8).reshape(160, 160)
    square = get_square_img(0, 0, img, trim_ratio=0.15)
    assert square.shape == (14, 14)
    full = get_square_img(0, 0, img, trim_ratio=0.0)
    assert full.shape == (20, 20)


def test_square_center():
    region = (100, 200, 900, 1000)
    assert square_center("e2", True, region) == (550, 850)
    assert square_center("a1", True, region) == (150, 950)
