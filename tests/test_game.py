import chess
import numpy as np
import pytest

from chessbot.game import BoardNotFound, GameSession
from chessbot.vision.position import board_to_grid, start_grid
from fakes import FakeBook, FakeCapturer, FakeEngine, FakeMouse, FakeRecognizer


def make_session(grid, turn_arg=None, confidence=0.99, confirm=None, engine=None):
    prompts = []

    def recording_confirm(prompt):
        prompts.append(prompt)
        return confirm if confirm is not None else "w"

    session = GameSession(
        capturer=FakeCapturer(np.zeros((512, 512, 3), np.uint8)),
        engine=engine or FakeEngine(["e2e4"]),
        book=FakeBook(),
        recognizer=FakeRecognizer(grid, confidence),
        mouse=FakeMouse(),
        turn_arg=turn_arg,
        confirm=recording_confirm,
        log=lambda *a: None,
    )
    session.region = (0, 0, 512, 512)
    return session, prompts


def test_read_position_startpos_needs_no_prompt():
    session, prompts = make_session(start_grid(True))
    board, white_at_bottom = session.read_position()
    assert board.fen() == chess.STARTING_FEN
    assert white_at_bottom is True
    assert prompts == []


def test_read_position_startpos_black_perspective():
    session, _ = make_session(start_grid(False))
    board, white_at_bottom = session.read_position()
    assert board.fen() == chess.STARTING_FEN
    assert white_at_bottom is False


def test_read_position_midgame_uses_turn_arg():
    ref = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3")
    session, prompts = make_session(board_to_grid(ref, True), turn_arg=chess.WHITE)
    board, _ = session.read_position()
    assert board.board_fen() == ref.board_fen()
    assert board.turn == chess.WHITE
    assert prompts == []


def test_read_position_midgame_prompts_for_turn():
    ref = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3")
    session, prompts = make_session(board_to_grid(ref, True), confirm="b")
    board, _ = session.read_position()
    assert board.turn == chess.BLACK
    assert len(prompts) == 1


def test_read_position_low_confidence_asks_confirmation():
    session, prompts = make_session(start_grid(True), confidence=0.5, confirm="y")
    session.read_position()
    assert any("Continue" in p for p in prompts)


def test_read_position_low_confidence_abort():
    session, _ = make_session(start_grid(True), confidence=0.5, confirm="n")
    with pytest.raises(SystemExit):
        session.read_position()


def test_locate_raises_after_attempts():
    session, _ = make_session(start_grid(True))
    with pytest.raises(BoardNotFound):
        session.locate(attempts=2, wait=0.0)


def test_choose_move_prefers_book():
    session, _ = make_session(start_grid(True), engine=FakeEngine(["d2d4"]))
    session.book = FakeBook("e2e4")
    assert session._choose_move(chess.Board(), total_moves=0) == "e2e4"


def test_choose_move_uses_engine_when_book_empty():
    engine = FakeEngine(["d2d4"])
    session, _ = make_session(start_grid(True), engine=engine)
    assert session._choose_move(chess.Board(), total_moves=0) == "d2d4"
    assert engine.calls[0]["move_time"] == 2.0


def test_choose_move_timing_engine_mode_caps_movetime():
    engine = FakeEngine(["d2d4"])
    session, _ = make_session(start_grid(True), engine=engine)
    session.timing_mode = "engine"
    session.timing_window = (0.0, 0.0)
    session._choose_move(chess.Board(), total_moves=6)
    assert engine.calls[0]["move_time"] == 0.0
