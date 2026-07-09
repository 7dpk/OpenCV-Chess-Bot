import chess
import cv2
import numpy as np
import pytest

from chessbot.game import BoardNotFound, GameSession
from chessbot.vision.position import board_to_grid, start_grid
from fakes import (
    FakeBook,
    FakeCapturer,
    FakeEngine,
    FakeMouse,
    FakeRecognizer,
    SequenceCapturer,
    SequenceRecognizer,
)
from helpers import occupied_squares, render_fake_board


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


def test_read_position_low_confidence_names_squares():
    session, _ = make_session(start_grid(True), confidence=0.5, confirm="y")
    logs = []
    session.log = lambda *a: logs.append(" ".join(str(x) for x in a))
    session.read_position()
    text = "\n".join(logs)
    assert "64 square" in text
    assert "a1=0.50" in text


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


def _grid_missing_white_king():
    grid = [row[:] for row in start_grid(True)]
    grid[7][4] = "empty"  # e1, where the white king starts
    return grid


def test_resync_retries_past_invalid_grid(monkeypatch):
    monkeypatch.setattr("chessbot.game.time.sleep", lambda *_: None)
    good_grid = start_grid(True)
    session, _ = make_session(good_grid)
    session.recognizer = SequenceRecognizer(
        [(_grid_missing_white_king(), 0.99), (good_grid, 0.99)]
    )
    board = session._resync()
    assert board.board_fen() == chess.Board().board_fen()


def test_resync_raises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr("chessbot.game.time.sleep", lambda *_: None)
    session, _ = make_session(start_grid(True))
    session.recognizer = SequenceRecognizer([(_grid_missing_white_king(), 0.99)])
    with pytest.raises(RuntimeError):
        session._resync()


def test_resync_adopts_best_effort_read_above_soft_floor(monkeypatch):
    """A persistent single low-confidence square must not kill the game: after
    exhausting attempts, a valid read above the soft floor is adopted."""
    monkeypatch.setattr("chessbot.game.time.sleep", lambda *_: None)
    session, _ = make_session(start_grid(True))
    session.recognizer = SequenceRecognizer([(start_grid(True), 0.85)])
    board = session._resync(attempts=3)
    assert board.board_fen() == chess.Board().board_fen()


def test_resync_never_adopts_below_soft_floor(monkeypatch):
    monkeypatch.setattr("chessbot.game.time.sleep", lambda *_: None)
    session, _ = make_session(start_grid(True))
    session.recognizer = SequenceRecognizer([(start_grid(True), 0.5)])
    with pytest.raises(RuntimeError):
        session._resync(attempts=3)


def test_await_opponent_rejects_ambiguous_candidates(monkeypatch):
    monkeypatch.setattr("chessbot.game.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "chessbot.game.move_detect.find_candidate_moves",
        lambda *a, **k: ["a1a2", "a1a3"],
    )
    session, _ = make_session(start_grid(True))
    session.capturer = FakeCapturer(np.full((512, 512, 3), 200, np.uint8))
    old = np.full((512, 512), 50, np.uint8)
    result = session._await_opponent(chess.Board(), old)
    assert result[0] == "resync"


# --- FIX 1 (I1): resync must not blindly adopt turn when placement is unchanged ---


def test_adopt_resync_keeps_current_board_when_placement_matches():
    session, _ = make_session(start_grid(True))
    current = chess.Board()  # turn == WHITE
    resynced = chess.Board()
    resynced.turn = chess.BLACK  # same placement, but a false-positive flipped turn
    result = session._adopt_resync(current, resynced)
    assert result is current
    assert result.turn == chess.WHITE


def test_adopt_resync_uses_resynced_board_when_placement_differs():
    session, _ = make_session(start_grid(True))
    current = chess.Board()
    resynced = chess.Board()
    resynced.push(chess.Move.from_uci("e2e4"))
    result = session._adopt_resync(current, resynced)
    assert result is resynced
    assert result.board_fen() == resynced.board_fen()


# --- FIX 2 (I3): periodic recognizer check during the first _await_opponent loop ---


def test_await_opponent_periodic_recognizer_check_detects_drift(monkeypatch):
    monkeypatch.setattr("chessbot.game.time.sleep", lambda *_: None)
    board = chess.Board()
    start_frame = cv2.cvtColor(render_fake_board(occupied_squares(board)), cv2.COLOR_GRAY2BGR)

    drifted = chess.Board()
    drifted.push(chess.Move.from_uci("e2e4"))

    session, _ = make_session(start_grid(True))
    session.RECOGNIZER_CHECK_INTERVAL = 3
    session.capturer = FakeCapturer(start_frame)
    session.recognizer = FakeRecognizer(board_to_grid(drifted, True), 0.99)

    old = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
    result = session._await_opponent(board, old)
    assert result[0] == "resync"


def test_await_opponent_periodic_recognizer_check_matching_grid_no_spurious_resync(monkeypatch):
    monkeypatch.setattr("chessbot.game.time.sleep", lambda *_: None)
    board = chess.Board()
    start_frame = cv2.cvtColor(render_fake_board(occupied_squares(board)), cv2.COLOR_GRAY2BGR)

    after = chess.Board()
    after.push_uci("e2e4")
    after_frame = cv2.cvtColor(render_fake_board(occupied_squares(after)), cv2.COLOR_GRAY2BGR)

    session, _ = make_session(start_grid(True))
    session.RECOGNIZER_CHECK_INTERVAL = 3
    # Enough static frames for a couple of periodic checks (grid matches, so no
    # resync fires) before the frame changes to reflect a real move.
    session.capturer = SequenceCapturer([start_frame] * 5 + [after_frame] * 2)
    session.recognizer = FakeRecognizer(start_grid(True), 0.99)

    old = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
    result = session._await_opponent(board, old)
    assert result[0] == "move"
    assert result[1] == "e2e4"
