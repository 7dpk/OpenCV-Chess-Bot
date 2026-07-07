import itertools

import numpy as np


class FakeCapturer:
    name = "fake"
    scale = 1.0

    def __init__(self, frames):
        self._frames = itertools.cycle(frames) if isinstance(frames, list) else itertools.cycle([frames])

    def grab(self, region=None):
        frame = next(self._frames)
        if region is not None:
            left, top, right, bottom = region
            frame = frame[top:bottom, left:right]
        return np.ascontiguousarray(frame)

    def close(self):
        pass


class FakeRecognizer:
    def __init__(self, grid, confidence=0.99):
        self.grid = grid
        self.confidence = confidence

    def classify_squares(self, board_img):
        return self.grid, np.full((8, 8), self.confidence, np.float32)


class SequenceRecognizer:
    """Stateful recognizer fake: replays a fixed sequence of (grid, confidence) results,
    then repeats the last one indefinitely."""

    def __init__(self, results):
        self._results = list(results)
        self._index = 0

    def classify_squares(self, board_img):
        grid, confidence = self._results[min(self._index, len(self._results) - 1)]
        self._index += 1
        return grid, np.full((8, 8), confidence, np.float32)


class FakeEngine:
    def __init__(self, moves):
        self.moves = list(moves)
        self.calls = []

    def best_move(self, board, *, depth=None, move_time=None):
        self.calls.append({"depth": depth, "move_time": move_time})
        return self.moves.pop(0)

    def close(self):
        pass


class FakeBook:
    def __init__(self, move=None):
        self.move = move

    def probe(self, board):
        return self.move

    def close(self):
        pass


class FakeMouse:
    def __init__(self):
        self.moves = []

    def play_move(self, uci, white_at_bottom, region):
        self.moves.append(uci)

    def click_square(self, name, white_at_bottom, region):
        pass
