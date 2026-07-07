import chess

from chessbot.vision.move_detect import (
    board_changed,
    detect_stealth_move,
    find_candidate_moves,
    is_empty,
)
from chessbot.vision.squares import get_square_img
from helpers import occupied_squares, render_fake_board


def test_is_empty_on_fake_board():
    img = render_fake_board({"e2"})
    from chessbot.vision.squares import square_name_to_row_col

    r, c = square_name_to_row_col("e2", True)
    assert not is_empty(get_square_img(r, c, img))
    r, c = square_name_to_row_col("e4", True)
    assert is_empty(get_square_img(r, c, img))


def test_board_changed():
    old = render_fake_board({"e2"})
    same = render_fake_board({"e2"})
    new = render_fake_board({"e4"})
    assert not board_changed(old, same)
    assert board_changed(old, new)


def test_find_candidate_moves_simple_pawn_push():
    board = chess.Board()
    old = render_fake_board(occupied_squares(board))
    board_after = chess.Board()
    board_after.push_uci("e2e4")
    new = render_fake_board(occupied_squares(board_after))
    assert find_candidate_moves(old, new, True, board) == ["e2e4"]


def test_find_candidate_moves_black_perspective():
    board = chess.Board()
    old = render_fake_board(occupied_squares(board), white_at_bottom=False)
    after = chess.Board()
    after.push_uci("g1f3")
    new = render_fake_board(occupied_squares(after), white_at_bottom=False)
    assert find_candidate_moves(old, new, False, board) == ["g1f3"]


def test_find_candidate_moves_castling():
    board = chess.Board("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    old = render_fake_board(occupied_squares(board))
    after = board.copy()
    after.push_uci("e1g1")
    new = render_fake_board(occupied_squares(after))
    assert find_candidate_moves(old, new, True, board) == ["e1g1"]


def test_find_candidate_moves_promotion():
    board = chess.Board("8/4P3/8/8/8/2k5/8/2K5 w - - 0 1")
    old = render_fake_board(occupied_squares(board))
    after = board.copy()
    after.push_uci("e7e8q")
    new = render_fake_board(occupied_squares(after))
    assert find_candidate_moves(old, new, True, board) == ["e7e8q"]


def test_find_candidate_moves_unresolvable_returns_empty():
    board = chess.Board()
    old = render_fake_board(occupied_squares(board))
    new = render_fake_board(set())
    assert find_candidate_moves(old, new, True, board) == []


def test_detect_stealth_move():
    board = chess.Board()
    after = chess.Board()
    after.push_uci("e2e4")
    img = render_fake_board(occupied_squares(after))
    assert detect_stealth_move(img, board, True) == "e2e4"
    in_sync = render_fake_board(occupied_squares(board))
    assert detect_stealth_move(in_sync, board, True) is None
