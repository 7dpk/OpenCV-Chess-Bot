import chess
import pytest

from chessbot.vision.position import (
    CLASSES,
    board_to_grid,
    grid_to_board,
    infer_white_at_bottom,
    label_to_piece,
    screen_to_square_name,
    start_grid,
)


def test_screen_to_square_name():
    assert screen_to_square_name(0, 0, True) == "a8"
    assert screen_to_square_name(7, 4, True) == "e1"
    assert screen_to_square_name(0, 0, False) == "h1"
    assert screen_to_square_name(7, 7, False) == "a8"


def test_classes_order():
    assert CLASSES[0] == "empty"
    assert len(CLASSES) == 13
    assert CLASSES[1] == "wP" and CLASSES[6] == "wK" and CLASSES[12] == "bK"


def test_label_to_piece():
    assert label_to_piece("empty") is None
    assert label_to_piece("wP") == chess.Piece.from_symbol("P")
    assert label_to_piece("bQ") == chess.Piece.from_symbol("q")


def test_start_grid_white_bottom():
    grid = start_grid(True)
    assert grid[0] == ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"]
    assert grid[6] == ["wP"] * 8
    assert grid[3] == ["empty"] * 8


def test_start_grid_black_bottom():
    grid = start_grid(False)
    assert grid[7] == ["bR", "bN", "bB", "bK", "bQ", "bB", "bN", "bR"]
    assert grid[0] == ["wR", "wN", "wB", "wK", "wQ", "wB", "wN", "wR"]


def test_grid_to_board_startpos_both_orientations():
    for white_at_bottom in (True, False):
        board = grid_to_board(start_grid(white_at_bottom), white_at_bottom, chess.WHITE)
        assert board.fen() == chess.STARTING_FEN


def test_board_to_grid_roundtrip():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3")
    for white_at_bottom in (True, False):
        grid = board_to_grid(board, white_at_bottom)
        rebuilt = grid_to_board(grid, white_at_bottom, chess.WHITE)
        assert rebuilt.board_fen() == board.board_fen()


def test_castling_heuristic():
    board = chess.Board("r3k3/8/8/8/8/8/8/4K2R w - - 0 1")
    grid = board_to_grid(board, True)
    rebuilt = grid_to_board(grid, True, chess.WHITE)
    assert rebuilt.has_kingside_castling_rights(chess.WHITE)
    assert rebuilt.has_queenside_castling_rights(chess.BLACK)
    assert not rebuilt.has_queenside_castling_rights(chess.WHITE)
    assert not rebuilt.has_kingside_castling_rights(chess.BLACK)


def test_castling_not_granted_off_home_squares():
    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    grid = board_to_grid(board, True)
    rebuilt = grid_to_board(grid, True, chess.BLACK)
    assert rebuilt.has_queenside_castling_rights(chess.WHITE)
    assert rebuilt.turn == chess.BLACK


def test_infer_white_at_bottom():
    assert infer_white_at_bottom(start_grid(True)) is True
    assert infer_white_at_bottom(start_grid(False)) is False
    lone = [["empty"] * 8 for _ in range(8)]
    lone[7][4] = "wK"
    lone[0][4] = "bK"
    assert infer_white_at_bottom(lone) is True


def test_grid_to_board_rejects_missing_king():
    grid = [["empty"] * 8 for _ in range(8)]
    grid[7][4] = "wK"
    with pytest.raises(ValueError, match="king"):
        grid_to_board(grid, True, chess.WHITE)


def test_grid_to_board_rejects_back_rank_pawn():
    grid = start_grid(True)
    grid[0][0] = "wP"
    with pytest.raises(ValueError, match="pawn"):
        grid_to_board(grid, True, chess.WHITE)
