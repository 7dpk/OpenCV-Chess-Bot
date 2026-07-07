import time

import chess
import cv2

from .capture.base import Region
from .engine.uci import sample_timing
from .vision import board_detect, move_detect
from .vision.position import grid_to_board, infer_white_at_bottom, start_grid
from .vision.squares import get_square_img, square_name_to_row_col


class BoardNotFound(RuntimeError):
    pass


class GameSession:
    def __init__(
        self,
        capturer,
        engine,
        book,
        recognizer,
        mouse,
        *,
        depth_mode: bool = False,
        depth: int = 12,
        move_time: float = 2.0,
        timing_mode: str | None = None,
        timing_window: tuple[float, float] | None = None,
        turn_arg: chess.Color | None = None,
        confidence_floor: float = 0.90,
        confirm=input,
        log=print,
    ):
        self.capturer = capturer
        self.engine = engine
        self.book = book
        self.recognizer = recognizer
        self.mouse = mouse
        self.depth_mode = depth_mode
        self.depth = depth
        self.move_time = move_time
        self.timing_mode = timing_mode
        self.timing_window = timing_window
        self.turn_arg = turn_arg
        self.confidence_floor = confidence_floor
        self.confirm = confirm
        self.log = log
        self.region: Region | None = None
        self.white_at_bottom = True
        self._depth_control = 32

    def locate(self, attempts: int = 30, wait: float = 1.0) -> Region:
        for _ in range(attempts):
            region = board_detect.locate_board(self.capturer.grab())
            if region is not None:
                self.region = region
                return region
            self.log("Board not found on screen, retrying...")
            time.sleep(wait)
        raise BoardNotFound("could not find a chessboard on screen; make sure it is fully visible")

    def read_position(self) -> tuple[chess.Board, bool]:
        img = self.capturer.grab(self.region)
        grid, confidence = self.recognizer.classify_squares(img)
        white_at_bottom = infer_white_at_bottom(grid)
        if grid in (start_grid(True), start_grid(False)):
            turn = chess.WHITE
        elif self.turn_arg is not None:
            turn = self.turn_arg
        else:
            answer = self.confirm("Whose turn is it? [w/b]: ").strip().lower()
            turn = chess.WHITE if answer.startswith("w") else chess.BLACK
        board = grid_to_board(grid, white_at_bottom, turn)
        if float(confidence.min()) < self.confidence_floor:
            self.log(f"Low recognition confidence ({confidence.min():.2f}). Detected position:")
            self.log(board.fen())
            answer = self.confirm("Continue with this position? [Y/n]: ")
            if answer.strip().lower().startswith("n"):
                raise SystemExit("aborted: position not confirmed")
        return board, white_at_bottom

    def run(self) -> None:
        if self.region is None:
            self.locate()
        self.log(f"Board region: {self.region}")
        board, self.white_at_bottom = self.read_position()
        our_color = chess.WHITE if self.white_at_bottom else chess.BLACK
        self.log(f"Playing {chess.COLOR_NAMES[our_color]}; position: {board.fen()}")
        old = self._grab_gray()
        total_moves = 0
        while not board.is_game_over():
            if board.turn == our_color:
                move = self._choose_move(board, total_moves)
                old = self._execute_move(move)
                board.push(chess.Move.from_uci(move))
                total_moves += 1
                self.log(f"We played {move}")
                continue
            result = self._await_opponent(board, old)
            if result[0] == "resync":
                board = result[1]
                old = self._grab_gray()
                self.log(f"Re-synced position: {board.fen()}")
                continue
            move, old = result[1], result[2]
            board.push(chess.Move.from_uci(move))
            self.log(f"They played {move}")
        self.log(f"Game over: {board.result()}")

    def _grab_gray(self):
        return cv2.cvtColor(self.capturer.grab(self.region), cv2.COLOR_BGR2GRAY)

    def _choose_move(self, board: chess.Board, total_moves: int) -> str:
        chosen_t = sample_timing(self.timing_window) if self.timing_mode else 0.0
        start_ts = time.time()
        move = self.book.probe(board) if total_moves < 5 else None
        if move is None:
            if self.depth_mode:
                move = self.engine.best_move(board, depth=self.depth)
                self._maybe_deepen(board)
            else:
                if self.timing_mode in ("engine", "both"):
                    move = self.engine.best_move(board, move_time=chosen_t)
                else:
                    move = self.engine.best_move(board, move_time=self.move_time)
        if self.timing_mode in ("delay", "both"):
            remaining = chosen_t - (time.time() - start_ts)
            if remaining > 0:
                time.sleep(remaining)
        return move

    def _maybe_deepen(self, board: chess.Board) -> None:
        pieces = len(board.piece_map())
        if pieces < self._depth_control and pieces % 2 == 0:
            self._depth_control = pieces
            self.depth += 1
            self.log(f"Depth increased to {self.depth}")

    def _execute_move(self, move: str):
        self.mouse.play_move(move, self.white_at_bottom, self.region)
        deadline = time.time() + 5.0
        start_rc = square_name_to_row_col(move[:2], self.white_at_bottom)
        end_rc = square_name_to_row_col(move[2:4], self.white_at_bottom)
        while time.time() < deadline:
            img = self._grab_gray()
            if move_detect.is_empty(get_square_img(*start_rc, img)) and not move_detect.is_empty(
                get_square_img(*end_rc, img)
            ):
                return img
            time.sleep(0.05)
        self.log("Move did not appear; if a promotion dialog is open, finish it manually.")
        self.confirm("Press Enter once the move is on the board: ")
        return self._grab_gray()

    def _await_opponent(self, board: chess.Board, old):
        while True:
            time.sleep(0.02)
            new = self._grab_gray()
            if move_detect.board_changed(old, new):
                break
            stealth = move_detect.detect_stealth_move(new, board, self.white_at_bottom)
            if stealth:
                return ("move", stealth, new)
        stable_misses = 0
        while True:
            time.sleep(0.02)
            current = self._grab_gray()
            if move_detect.board_changed(new, current):
                new = current
                continue
            moves = move_detect.find_candidate_moves(old, current, self.white_at_bottom, board)
            if len(moves) == 1:
                return ("move", moves[0], current)
            stable_misses += 1
            if stable_misses > 40:
                return ("resync", self._resync())

    def _resync(self, attempts: int = 30, wait: float = 0.5) -> chess.Board:
        our_color = chess.WHITE if self.white_at_bottom else chess.BLACK
        for _ in range(attempts):
            img = self.capturer.grab(self.region)
            grid, confidence = self.recognizer.classify_squares(img)
            if float(confidence.min()) < self.confidence_floor:
                self.log(f"Re-sync attempt failed: confidence {confidence.min():.2f} below floor")
            else:
                try:
                    return grid_to_board(grid, self.white_at_bottom, our_color)
                except ValueError as exc:
                    self.log(f"Re-sync attempt failed: {exc}")
            time.sleep(wait)
        raise RuntimeError("could not re-sync position from screen")
