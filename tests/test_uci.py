import random
import shutil

import chess
import pytest

from chessbot.config import Settings
from chessbot.engine.uci import Book, EngineClient, find_engine, parse_tc, sample_timing


def test_parse_tc_minutes():
    assert parse_tc("40/5m") == pytest.approx(7.5)
    assert parse_tc("40/5") == pytest.approx(7.5)


def test_parse_tc_seconds_hours_ms():
    assert parse_tc("40/300s") == pytest.approx(7.5)
    assert parse_tc("40/0.5h") == pytest.approx(45.0)
    assert parse_tc("40/400000ms") == pytest.approx(10.0)


@pytest.mark.parametrize("bad", ["", "5m", "30/5m", "40/xyz", "40/-3s", "40/0"])
def test_parse_tc_rejects(bad):
    with pytest.raises(ValueError):
        parse_tc(bad)


def test_sample_timing_within_window():
    rng = random.Random(7)
    for _ in range(50):
        t = sample_timing((1.5, 3.0), rng)
        assert 1.5 <= t <= 3.0


def test_book_probe_startpos():
    book = Book(Settings().book_path)
    move = book.probe(chess.Board())
    assert move is not None
    assert chess.Move.from_uci(move) in chess.Board().legal_moves
    book.close()


def test_book_probe_unknown_position_returns_none():
    book = Book(Settings().book_path)
    board = chess.Board("k7/8/8/8/8/8/8/K7 w - - 0 1")
    assert book.probe(board) is None
    book.close()


def test_find_engine_missing():
    with pytest.raises(FileNotFoundError, match="--engine"):
        find_engine("/nonexistent/engine/path")


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="stockfish not installed")
def test_engine_client_plays_legal_move():
    client = EngineClient(find_engine(None), threads=1, hash_mb=64)
    board = chess.Board()
    move = client.best_move(board, move_time=0.05)
    assert chess.Move.from_uci(move) in board.legal_moves
    move = client.best_move(board, depth=4)
    assert chess.Move.from_uci(move) in board.legal_moves
    client.close()
