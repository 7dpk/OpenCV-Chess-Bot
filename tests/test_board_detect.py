import numpy as np

from chessbot.vision.board_detect import locate_board


def synthetic_screen(board_size=480, offset=(272, 144), canvas=(768, 1024)):
    img = np.full((canvas[0], canvas[1], 3), 255, np.uint8)
    left, top = offset
    sq = board_size // 8
    for r in range(8):
        for c in range(8):
            val = 200 if (r + c) % 2 == 0 else 80
            img[top + r * sq:top + (r + 1) * sq, left + c * sq:left + (c + 1) * sq] = val
    return img


def test_locate_board_finds_region():
    img = synthetic_screen()
    region = locate_board(img)
    assert region is not None
    left, top, right, bottom = region
    assert abs(left - 272) <= 12
    assert abs(top - 144) <= 12
    assert abs(right - 752) <= 12
    assert abs(bottom - 624) <= 12


def test_locate_board_other_offset_and_size():
    img = synthetic_screen(board_size=400, offset=(50, 90), canvas=(600, 800))
    region = locate_board(img)
    assert region is not None
    left, top, right, bottom = region
    assert abs(left - 50) <= 12 and abs(top - 90) <= 12
    assert abs(right - 450) <= 12 and abs(bottom - 490) <= 12


def test_locate_board_none_on_blank_screen():
    img = np.full((600, 800, 3), 128, np.uint8)
    assert locate_board(img) is None
