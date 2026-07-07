import socket
import threading

import chess
import pytest

from chessbot.engine.remote import RemoteEngine
from chessbot.engine.server import compute_move, serve
from fakes import FakeEngine


def test_compute_move_time_mode():
    engine = FakeEngine(["e2e4"])
    move = compute_move(f"{chess.STARTING_FEN},time,0.5", engine)
    assert move == "e2e4"
    assert engine.calls[0]["move_time"] == 0.5


def test_compute_move_depth_mode():
    engine = FakeEngine(["g1f3"])
    move = compute_move(f"{chess.STARTING_FEN},depth,8", engine)
    assert move == "g1f3"
    assert engine.calls[0]["depth"] == 8


def test_compute_move_rejects_garbage():
    with pytest.raises(ValueError):
        compute_move("not-a-fen", FakeEngine([]))


def test_remote_engine_round_trip():
    engine = FakeEngine(["e2e4", "d2d4"])
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    thread = threading.Thread(target=serve, args=(engine, "127.0.0.1", port), daemon=True)
    thread.start()
    remote = RemoteEngine("127.0.0.1", port)
    board = chess.Board()
    assert remote.best_move(board, move_time=0.1) == "e2e4"
    assert remote.best_move(board, depth=6) == "d2d4"
    remote.close()
