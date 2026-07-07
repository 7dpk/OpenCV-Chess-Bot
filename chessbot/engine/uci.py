import random
import shutil
from pathlib import Path

import chess
import chess.engine
import chess.polyglot


def parse_tc(tc: str) -> float:
    """'40/Xu' -> per-move seconds for 40 moves. Units: ms, s, m (default), h."""
    s = tc.strip().lower()
    if "/" not in s:
        raise ValueError("time control must look like 40/5m, 40/300s, 40/0.5h")
    left, right = s.split("/", 1)
    try:
        moves = int(left)
    except ValueError:
        raise ValueError("left side must be an integer move count") from None
    if moves != 40:
        raise ValueError("only 40/X is supported")
    units = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    unit = "m"
    value_str = right
    for suffix in ("ms", "s", "m", "h"):
        if right.endswith(suffix):
            unit = suffix
            value_str = right[: -len(suffix)]
            break
    try:
        total = float(value_str)
    except ValueError:
        raise ValueError("right side must be a number") from None
    if total <= 0:
        raise ValueError("total time must be positive")
    return total * units[unit] / 40.0


def sample_timing(window: tuple[float, float], rng=random) -> float:
    return rng.uniform(window[0], window[1])


def find_engine(explicit: str | None) -> str:
    if explicit:
        if Path(explicit).is_file():
            return explicit
        raise FileNotFoundError(f"engine not found at {explicit!r}; pass a valid --engine path")
    found = shutil.which("stockfish")
    if found:
        return found
    raise FileNotFoundError("no UCI engine found on PATH; install stockfish or pass --engine")


class Book:
    def __init__(self, path: str | Path):
        self._reader = chess.polyglot.open_reader(path)

    def probe(self, board: chess.Board) -> str | None:
        try:
            return str(next(self._reader.find_all(board)).move)
        except StopIteration:
            return None

    def close(self) -> None:
        self._reader.close()


class EngineClient:
    def __init__(self, path: str, threads: int, hash_mb: int, skill_level: int = 20):
        self._engine = chess.engine.SimpleEngine.popen_uci(path)
        options = {"Hash": hash_mb, "Threads": threads}
        if "Skill Level" in self._engine.options:
            options["Skill Level"] = skill_level
        self._engine.configure(options)

    def best_move(self, board: chess.Board, *, depth: int | None = None, move_time: float | None = None) -> str:
        if depth is not None:
            limit = chess.engine.Limit(depth=depth)
        else:
            limit = chess.engine.Limit(time=move_time if move_time and move_time > 0 else 0.05)
        return str(self._engine.play(board, limit, ponder=False).move)

    def close(self) -> None:
        self._engine.quit()
