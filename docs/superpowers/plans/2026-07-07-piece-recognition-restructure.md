# Piece Recognition + Repo Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the bot into an installable `chessbot` package, add CNN-based piece recognition (synthetic training data from lichess/chess.com piece sets, ONNX inference via cv2.dnn) so the bot can join a game from any position, and make screen capture native and fast on Windows (dxcam) and macOS (mss).

**Architecture:** Runtime package `chessbot/` with capture / vision / engine / control layers wired by `game.py` and a `chessbot` CLI. A separate `training/` directory (dev-time only, torch behind an optional extra) synthesizes labeled square crops from downloaded piece assets, trains a ~55K-param CNN, and exports ONNX committed at `models/piece_classifier.onnx`. The proven diff-based move detection keeps driving the game loop; the recognizer builds the initial FEN and re-syncs the board whenever diffing fails (which previously killed the process).

**Tech Stack:** Python ≥3.10, opencv-python, numpy, python-chess (`chess`), pyautogui, mss, dxcam (Windows extra); torch + cairosvg + pillow + requests (training extra); pytest.

**Spec:** `docs/superpowers/specs/2026-07-07-piece-recognition-restructure-design.md`

## Global Constraints

- Python `>=3.10`. Dev machine is macOS (Python 3.11); Windows-only code paths must be import-guarded and test-skipped off-Windows.
- Runtime dependencies are exactly: `opencv-python`, `numpy`, `chess`, `pyautogui`, `mss`. `dxcam` only via the `[windows]` extra. `torch`, `cairosvg`, `pillow`, `requests` only via the `[training]` extra. No PIL, no `keyboard`, no `d3dshot` at runtime.
- Image convention: `np.uint8` BGR everywhere at runtime. `Region = (left, top, right, bottom)` in capture pixels.
- Class order (single source of truth `chessbot/vision/position.py`):
  `CLASSES = ["empty", "wP", "wN", "wB", "wR", "wQ", "wK", "bP", "bN", "bB", "bR", "bQ", "bK"]`
- Model contract: input 48×48 BGR, preprocessing = pixel/255.0 only (no mean subtraction, no RB swap), output raw logits shape `(N, 13)`, ONNX opset 12, dynamic batch axis, committed at `models/piece_classifier.onnx`.
- Screen grid convention: `grid[row][col]`, row 0 = top of screen. Orientation handled explicitly via `white_at_bottom: bool`.
- Commit messages: conventional style (`feat:`, `refactor:`, `test:`, `docs:`, `chore:`). Never mention AI/assistant/Claude, no `Co-Authored-By` lines.
- Comments: sparse — only for constraints the code cannot express. No narration comments.
- Never commit downloaded assets (`training/assets/`) or generated datasets (`training/dataset/`). The trained ONNX model and a few self-rendered lichess-based test fixtures ARE committed.
- `default.ini` stays at repo root with its legacy section names (`Engine Default Settings`, `Time Control`).
- Legacy scripts (`bot-offline.py`, `bot-online.py`, `bot_server.py`, `main.py`, `setup.py`, `test.py`) remain in the tree until Task 19 deletes them; they are Windows-only and already non-functional on this machine, so intermediate breakage of them is expected and fine.

## File Structure

```
pyproject.toml                          Task 1
chessbot/__init__.py                    Task 1
chessbot/config.py                      Task 7
chessbot/cli.py                         Task 13
chessbot/game.py                        Task 11
chessbot/capture/__init__.py            Task 6 (factory)
chessbot/capture/base.py                Task 6
chessbot/capture/mss_backend.py         Task 6
chessbot/capture/dxcam_backend.py       Task 6
chessbot/vision/__init__.py             Task 2
chessbot/vision/squares.py              Task 2
chessbot/vision/position.py             Task 3
chessbot/vision/board_detect.py         Task 4
chessbot/vision/move_detect.py          Task 5
chessbot/vision/recognizer.py           Task 9
chessbot/engine/__init__.py             Task 8
chessbot/engine/uci.py                  Task 8
chessbot/engine/server.py               Task 12
chessbot/engine/remote.py               Task 12
chessbot/control/__init__.py            Task 10
chessbot/control/mouse.py               Task 10
training/__init__.py                    Task 14
training/fetch_assets.py                Task 14
training/generate_dataset.py            Task 15
training/train.py                       Task 16
training/evaluate.py                    Task 17
models/piece_classifier.onnx            Task 18 (trained artifact)
assets/Performance.bin                  Task 1 (moved from root)
tests/helpers.py                        Task 5
tests/fakes.py                          Task 11
tests/test_squares.py                   Task 2
tests/test_position.py                  Task 3
tests/test_board_detect.py              Task 4
tests/test_move_detect.py               Task 5
tests/test_capture.py                   Task 6
tests/test_config.py                    Task 7
tests/test_uci.py                       Task 8
tests/test_recognizer.py                Task 9
tests/test_mouse.py                     Task 10
tests/test_game.py                      Task 11
tests/test_server.py                    Task 12
tests/test_cli.py                       Task 13
tests/test_fetch_assets.py              Task 14
tests/test_generate_dataset.py          Task 15
tests/test_training.py                  Task 16
tests/test_recognizer_model.py          Task 18 (integration, real model)
tests/fixtures/board_*.png              Task 18
README.md                               Task 19 (rewrite)
.gitignore                              Tasks 1, 18 (edits)
```

---

### Task 1: Package scaffold, pyproject, junk removal

**Files:**
- Create: `pyproject.toml`, `chessbot/__init__.py`, `chessbot/vision/__init__.py`, `chessbot/engine/__init__.py`, `chessbot/capture/__init__.py` (stub), `chessbot/control/__init__.py`, `tests/` dir
- Modify: `.gitignore`
- Move: `Performance.bin` → `assets/Performance.bin`
- Delete: `bot.cp38-win_amd64.pyd`, `chess_bot-main.zip`, `default.ini~`, `.default.ini.un~`, `.vscode/settings.json`

**Interfaces:**
- Produces: importable `chessbot` package, `pip install -e ".[dev]"` works, `pytest` runs (0 tests).

- [ ] **Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "chessbot"
version = "2.0.0"
description = "Screen-reading chess bot: board detection, piece recognition, UCI engine play"
requires-python = ">=3.10"
dependencies = [
    "opencv-python>=4.8",
    "numpy>=1.24",
    "chess>=1.10",
    "pyautogui>=0.9.54",
    "mss>=9.0",
]

[project.optional-dependencies]
windows = ["dxcam>=0.0.5; sys_platform == 'win32'"]
training = ["torch>=2.1", "cairosvg>=2.7", "pillow>=10.0", "requests>=2.31"]
dev = ["pytest>=8.0"]

[project.scripts]
chessbot = "chessbot.cli:main"

[tool.setuptools.packages.find]
include = ["chessbot*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package skeleton and clean junk**

```bash
mkdir -p chessbot/vision chessbot/engine chessbot/capture chessbot/control tests assets
touch chessbot/__init__.py chessbot/vision/__init__.py chessbot/engine/__init__.py \
      chessbot/capture/__init__.py chessbot/control/__init__.py
git mv Performance.bin assets/Performance.bin
git rm bot.cp38-win_amd64.pyd chess_bot-main.zip .vscode/settings.json
rm -f default.ini~ .default.ini.un~
```

- [ ] **Step 3: Update .gitignore**

Append to `.gitignore` (leave existing `models/` and `*.onnx` lines alone for now — Task 18 removes them when the model is committed):

```
# Downloaded training assets and generated datasets
training/assets/
training/dataset/
```

Also fix the three lines under `# ML datasets and training artifacts` that begin with a stray leading space (` captures/`, ` dataset/`, ` models/` — a leading space makes gitignore patterns silently non-functional): remove the leading space from each.

- [ ] **Step 4: Verify install and empty test run**

Run: `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" -q && .venv/bin/python -c "import chessbot; print('ok')" && .venv/bin/pytest`
Expected: `ok`, then pytest exits with "no tests ran".

All subsequent task commands assume this venv is active (`source .venv/bin/activate`).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: scaffold chessbot package, remove build junk"
```

---

### Task 2: Square geometry — `chessbot/vision/squares.py`

**Files:**
- Create: `chessbot/vision/squares.py`
- Test: `tests/test_squares.py`

**Interfaces:**
- Produces:
  - `get_square_img(row: int, col: int, img: np.ndarray, trim_ratio: float = 0.15) -> np.ndarray`
  - `row_col_to_square_name(row: int, col: int, white_at_bottom: bool) -> str`
  - `square_name_to_row_col(name: str, white_at_bottom: bool) -> tuple[int, int]`
  - `square_to_row_col(square: int, white_at_bottom: bool) -> tuple[int, int]` (chess.Square int)
  - `square_center(name: str, white_at_bottom: bool, region: tuple) -> tuple[int, int]` (pixel coords)

- [ ] **Step 1: Write the failing test**

`tests/test_squares.py`:

```python
import numpy as np
import chess

from chessbot.vision.squares import (
    get_square_img,
    row_col_to_square_name,
    square_center,
    square_name_to_row_col,
    square_to_row_col,
)


def test_row_col_to_square_name_white_bottom():
    assert row_col_to_square_name(7, 0, True) == "a1"
    assert row_col_to_square_name(0, 0, True) == "a8"
    assert row_col_to_square_name(6, 4, True) == "e2"


def test_row_col_to_square_name_black_bottom():
    assert row_col_to_square_name(0, 0, False) == "h1"
    assert row_col_to_square_name(7, 7, False) == "a8"
    assert row_col_to_square_name(1, 3, False) == "e2"


def test_square_name_roundtrip():
    for white_at_bottom in (True, False):
        for row in range(8):
            for col in range(8):
                name = row_col_to_square_name(row, col, white_at_bottom)
                assert square_name_to_row_col(name, white_at_bottom) == (row, col)


def test_square_to_row_col_matches_name_mapping():
    for white_at_bottom in (True, False):
        for sq in chess.SQUARES:
            row, col = square_to_row_col(sq, white_at_bottom)
            assert row_col_to_square_name(row, col, white_at_bottom) == chess.square_name(sq)


def test_get_square_img_shape_and_trim():
    img = np.arange(160 * 160, dtype=np.uint8).reshape(160, 160)
    square = get_square_img(0, 0, img, trim_ratio=0.15)
    assert square.shape == (14, 14)
    full = get_square_img(0, 0, img, trim_ratio=0.0)
    assert full.shape == (20, 20)


def test_square_center():
    region = (100, 200, 900, 1000)
    assert square_center("e2", True, region) == (550, 850)
    assert square_center("a1", True, region) == (150, 950)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_squares.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'chessbot.vision.squares'`

- [ ] **Step 3: Write the implementation**

`chessbot/vision/squares.py`:

```python
import chess
import numpy as np


def get_square_img(row: int, col: int, img: np.ndarray, trim_ratio: float = 0.15) -> np.ndarray:
    h, w = img.shape[:2]
    y0, y1 = int(row * h / 8), int((row + 1) * h / 8)
    x0, x1 = int(col * w / 8), int((col + 1) * w / 8)
    square = img[y0:y1, x0:x1]
    ty = int((y1 - y0) * trim_ratio)
    tx = int((x1 - x0) * trim_ratio)
    if ty and tx:
        return square[ty:-ty, tx:-tx]
    return square


def row_col_to_square_name(row: int, col: int, white_at_bottom: bool) -> str:
    if white_at_bottom:
        return chr(97 + col) + str(8 - row)
    return chr(97 + 7 - col) + str(row + 1)


def square_name_to_row_col(name: str, white_at_bottom: bool) -> tuple[int, int]:
    file_idx = ord(name[0]) - 97
    rank = int(name[1])
    if white_at_bottom:
        return 8 - rank, file_idx
    return rank - 1, 7 - file_idx


def square_to_row_col(square: int, white_at_bottom: bool) -> tuple[int, int]:
    rank = chess.square_rank(square)
    file_idx = chess.square_file(square)
    if white_at_bottom:
        return 7 - rank, file_idx
    return rank, 7 - file_idx


def square_center(name: str, white_at_bottom: bool, region: tuple) -> tuple[int, int]:
    left, top, right, bottom = region
    row, col = square_name_to_row_col(name, white_at_bottom)
    x = left + (col + 0.5) * (right - left) / 8
    y = top + (row + 0.5) * (bottom - top) / 8
    return int(x), int(y)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_squares.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add chessbot/vision/squares.py tests/test_squares.py
git commit -m "feat: square geometry helpers"
```

---

### Task 3: Position building — `chessbot/vision/position.py`

**Files:**
- Create: `chessbot/vision/position.py`
- Test: `tests/test_position.py`

**Interfaces:**
- Consumes: `square math from Task 2 conventions (grid[row][col], row 0 = top)`
- Produces:
  - `CLASSES: list[str]` (the 13-class order — training and recognizer import this)
  - `label_to_piece(label: str) -> chess.Piece | None`
  - `piece_to_label(piece: chess.Piece | None) -> str`
  - `infer_white_at_bottom(grid: list[list[str]]) -> bool`
  - `start_grid(white_at_bottom: bool) -> list[list[str]]`
  - `grid_to_board(grid, white_at_bottom: bool, turn: chess.Color) -> chess.Board` (raises `ValueError` on impossible positions)
  - `board_to_grid(board: chess.Board, white_at_bottom: bool) -> list[list[str]]`

- [ ] **Step 1: Write the failing test**

`tests/test_position.py`:

```python
import chess
import pytest

from chessbot.vision.position import (
    CLASSES,
    board_to_grid,
    grid_to_board,
    infer_white_at_bottom,
    label_to_piece,
    start_grid,
)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_position.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'chessbot.vision.position'`

- [ ] **Step 3: Write the implementation**

`chessbot/vision/position.py`:

```python
import chess

CLASSES = ["empty", "wP", "wN", "wB", "wR", "wQ", "wK", "bP", "bN", "bB", "bR", "bQ", "bK"]


def label_to_piece(label: str) -> chess.Piece | None:
    if label == "empty":
        return None
    symbol = label[1] if label[0] == "w" else label[1].lower()
    return chess.Piece.from_symbol(symbol)


def piece_to_label(piece: chess.Piece | None) -> str:
    if piece is None:
        return "empty"
    color = "w" if piece.color == chess.WHITE else "b"
    return color + piece.symbol().upper()


def _screen_to_square(row: int, col: int, white_at_bottom: bool) -> int:
    if white_at_bottom:
        return chess.square(col, 7 - row)
    return chess.square(7 - col, row)


def infer_white_at_bottom(grid: list[list[str]]) -> bool:
    score = 0
    for row in range(8):
        for col in range(8):
            label = grid[row][col]
            if label == "empty":
                continue
            bottom = row >= 4
            if label[0] == "w":
                score += 1 if bottom else -1
            else:
                score += -1 if bottom else 1
    return score >= 0


def board_to_grid(board: chess.Board, white_at_bottom: bool) -> list[list[str]]:
    return [
        [piece_to_label(board.piece_at(_screen_to_square(row, col, white_at_bottom))) for col in range(8)]
        for row in range(8)
    ]


def start_grid(white_at_bottom: bool) -> list[list[str]]:
    return board_to_grid(chess.Board(), white_at_bottom)


def _castling_rights(board: chess.Board) -> int:
    rights = 0
    if board.piece_at(chess.E1) == chess.Piece.from_symbol("K"):
        if board.piece_at(chess.H1) == chess.Piece.from_symbol("R"):
            rights |= chess.BB_H1
        if board.piece_at(chess.A1) == chess.Piece.from_symbol("R"):
            rights |= chess.BB_A1
    if board.piece_at(chess.E8) == chess.Piece.from_symbol("k"):
        if board.piece_at(chess.H8) == chess.Piece.from_symbol("r"):
            rights |= chess.BB_H8
        if board.piece_at(chess.A8) == chess.Piece.from_symbol("r"):
            rights |= chess.BB_A8
    return rights


def grid_to_board(grid: list[list[str]], white_at_bottom: bool, turn: chess.Color) -> chess.Board:
    board = chess.Board(None)
    for row in range(8):
        for col in range(8):
            piece = label_to_piece(grid[row][col])
            if piece is not None:
                board.set_piece_at(_screen_to_square(row, col, white_at_bottom), piece)
    board.turn = turn
    board.castling_rights = _castling_rights(board)
    _validate(board)
    return board


def _validate(board: chess.Board) -> None:
    for color in (chess.WHITE, chess.BLACK):
        if len(board.pieces(chess.KING, color)) != 1:
            raise ValueError("expected exactly one king per side; recognition is likely wrong")
    pawns = board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)
    if any(chess.square_rank(sq) in (0, 7) for sq in pawns):
        raise ValueError("pawn on a back rank; recognition is likely wrong")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_position.py -q`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add chessbot/vision/position.py tests/test_position.py
git commit -m "feat: grid-to-board position building with FEN heuristics"
```

---

### Task 4: Board detection port — `chessbot/vision/board_detect.py`

Port of the gradient/Hough detector from `bot-offline.py:22-271`. Changes from the original: PIL replaced with numpy slicing + `cv2.resize`, out-of-bounds crops handled with zero-padding, the row/column variable names unswapped (the original's `hough_gx` was actually per-row), the `dx`/`dy` mixup in sub-corner math fixed, and the inverted success check (`if not o: exit`) replaced with a sane `Region | None` API.

**Files:**
- Create: `chessbot/vision/board_detect.py`
- Test: `tests/test_board_detect.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (self-contained).
- Produces:
  - `Region = tuple[int, int, int, int]` (left, top, right, bottom)
  - `detect_chessboard_corners(gray: np.ndarray, noise_threshold: float = 8000) -> np.ndarray | None`
  - `locate_board(img_bgr: np.ndarray) -> Region | None` (thresholds, grays, detects, validates squareness)

- [ ] **Step 1: Write the failing test**

`tests/test_board_detect.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_board_detect.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'chessbot.vision.board_detect'`

- [ ] **Step 3: Write the implementation**

`chessbot/vision/board_detect.py`:

```python
import cv2
import numpy as np

Region = tuple[int, int, int, int]


def _get_all_sequences(seq, min_seq_len=7, err_px=5):
    """All subsequences with common spacing (within err_px) of length >= min_seq_len."""
    if len(seq) < min_seq_len:
        return []
    seqs = []
    for i in range(len(seq) - 1):
        for j in range(i + 1, len(seq)):
            duplicate = False
            for prev in seqs:
                for k in range(len(prev) - 1):
                    if seq[i] == prev[k] and seq[j] == prev[k + 1]:
                        duplicate = True
            if duplicate:
                continue
            d = seq[j] - seq[i]
            if d < err_px:
                continue
            s = [seq[i], seq[j]]
            n = s[-1] + d
            while np.abs(seq - n).min() < err_px:
                n = seq[np.abs(seq - n).argmin()]
                s.append(n)
                n = s[-1] + d
            if len(s) >= min_seq_len:
                seqs.append(np.array(s))
    return seqs


def _nonmax_suppress_1d(arr, winsize=5):
    out = arr.copy()
    for i in range(out.size):
        left = arr[max(0, i - winsize):i] if i > 0 else np.zeros(1)
        right = arr[i + 1:min(arr.size - 1, i + winsize)] if i < out.size - 2 else np.zeros(1)
        if left.size and arr[i] < left.max():
            out[i] = 0
        elif right.size and arr[i] <= right.max():
            out[i] = 0
    return out


def _trim_sequence(seq, vals):
    while len(seq) > 9:
        if vals[0] > vals[-1]:
            seq, vals = seq[:-1], vals[:-1]
        else:
            seq, vals = seq[1:], vals[1:]
    return seq, vals


def _crop_padded(img, x0, y0, x1, y1):
    h, w = img.shape
    out = np.zeros((y1 - y0, x1 - x0), img.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def detect_chessboard_corners(gray: np.ndarray, noise_threshold: float = 8000) -> np.ndarray | None:
    grad_rows, grad_cols = np.gradient(gray.astype(float))
    r_pos, r_neg = np.clip(grad_rows, 0, None), np.clip(-grad_rows, 0, None)
    c_pos, c_neg = np.clip(grad_cols, 0, None), np.clip(-grad_cols, 0, None)

    hough_rows = r_pos.sum(axis=1) * r_neg.sum(axis=1)
    hough_cols = c_pos.sum(axis=0) * c_neg.sum(axis=0)

    if min(hough_rows.std() / hough_rows.size, hough_cols.std() / hough_cols.size) < noise_threshold:
        return None

    hough_rows = _nonmax_suppress_1d(hough_rows) / hough_rows.max()
    hough_cols = _nonmax_suppress_1d(hough_cols) / hough_cols.max()
    hough_rows[hough_rows < 0.2] = 0
    hough_cols[hough_cols < 0.2] = 0

    lines_y = np.where(hough_rows)[0]
    lines_x = np.where(hough_cols)[0]
    vals_y = hough_rows[lines_y]
    vals_x = hough_cols[lines_x]

    seqs_y = _get_all_sequences(lines_y)
    seqs_x = _get_all_sequences(lines_x)
    if not seqs_y or not seqs_x:
        return None

    seqs_y_vals = [vals_y[[v in seq for v in lines_y]] for seq in seqs_y]
    seqs_x_vals = [vals_x[[v in seq for v in lines_x]] for seq in seqs_x]
    for i in range(len(seqs_y)):
        seqs_y[i], seqs_y_vals[i] = _trim_sequence(seqs_y[i], seqs_y_vals[i])
    for i in range(len(seqs_x)):
        seqs_x[i], seqs_x_vals[i] = _trim_sequence(seqs_x[i], seqs_x_vals[i])

    best_y = seqs_y[int(np.argmax([np.mean(v) for v in seqs_y_vals]))]
    best_x = seqs_x[int(np.argmax([np.mean(v) for v in seqs_x_vals]))]

    sub_y = [best_y[k:k + 7] for k in range(len(best_y) - 6)]
    sub_x = [best_x[k:k + 7] for k in range(len(best_x) - 6)]

    dy = int(np.median(np.diff(best_y)))
    dx = int(np.median(np.diff(best_x)))
    x0, y0 = int(best_x[0] - dx), int(best_y[0] - dy)
    x1, y1 = int(best_x[-1] + dx), int(best_y[-1] + dy)
    crop = _crop_padded(gray, x0, y0, x1, y1)

    quad = np.ones([8, 8])
    kernel = np.vstack([np.hstack([quad, -quad]), np.hstack([-quad, quad])])
    kernel = np.tile(kernel, (4, 4))
    kernel = kernel / np.linalg.norm(kernel)

    best_score = None
    final = None
    for sx in sub_x:
        for sy in sub_y:
            rx0 = int(sx[0] - dx - x0)
            ry0 = int(sy[0] - dy - y0)
            rx1 = int(sx[-1] + dx - x0)
            ry1 = int(sy[-1] + dy - y0)
            sub = crop[max(0, ry0):ry1, max(0, rx0):rx1]
            if sub.size == 0:
                continue
            sub = cv2.resize(sub, (64, 64), interpolation=cv2.INTER_NEAREST)
            score = abs(float(np.sum(kernel * sub)))
            if best_score is None or score > best_score:
                best_score = score
                final = np.array([rx0 + x0, ry0 + y0, rx1 + x0, ry1 + y0])
    return final


def locate_board(img_bgr: np.ndarray) -> Region | None:
    _, thresh = cv2.threshold(img_bgr, 169, 255, cv2.THRESH_BINARY)
    gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    corners = detect_chessboard_corners(gray)
    if corners is None:
        return None
    x0, y0, x1, y1 = (int(v) for v in corners)
    w, h = x1 - x0, y1 - y0
    if h <= 0 or w <= 0 or abs(1 - w / h) > 0.05:
        return None
    height, width = gray.shape
    return (max(0, x0), max(0, y0), min(width, x1), min(height, y1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_board_detect.py -q`
Expected: 3 passed. If the synthetic tests fail on tolerance, debug the port against `bot-offline.py` before touching thresholds — the algorithm is field-proven.

- [ ] **Step 5: Commit**

```bash
git add chessbot/vision/board_detect.py tests/test_board_detect.py
git commit -m "refactor: port board detection to package with cv2-only pipeline"
```

---

### Task 5: Move detection port — `chessbot/vision/move_detect.py`

Port of the diff-based move detection from `bot-offline.py:277-461`. Changes: no `os._exit`/`input()` in library code (returns `[]`/`None` instead — the game loop re-syncs), castling table instead of four if-blocks.

**Files:**
- Create: `chessbot/vision/move_detect.py`, `tests/helpers.py`
- Test: `tests/test_move_detect.py`

**Interfaces:**
- Consumes: `get_square_img`, `row_col_to_square_name`, `square_to_row_col`, `square_name_to_row_col` from Task 2.
- Produces:
  - `is_empty(square_img: np.ndarray) -> bool`
  - `board_changed(old: np.ndarray, new: np.ndarray) -> bool` (grayscale board images)
  - `find_candidate_moves(old, new, white_at_bottom: bool, board: chess.Board) -> list[str]` (legal UCI strings, `[]` when unresolvable)
  - `detect_stealth_move(img, board: chess.Board, white_at_bottom: bool) -> str | None`
  - tests helper: `render_fake_board(occupied: set[str], white_at_bottom=True, size=512) -> np.ndarray` in `tests/helpers.py`

- [ ] **Step 1: Write the test helper and failing test**

`tests/helpers.py`:

```python
import cv2
import numpy as np

from chessbot.vision.squares import square_name_to_row_col

LIGHT, DARK, PIECE = 200, 110, 30


def render_fake_board(occupied: set[str], white_at_bottom: bool = True, size: int = 512) -> np.ndarray:
    img = np.zeros((size, size), np.uint8)
    sq = size // 8
    for r in range(8):
        for c in range(8):
            img[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = LIGHT if (r + c) % 2 == 0 else DARK
    for name in occupied:
        r, c = square_name_to_row_col(name, white_at_bottom)
        center = (int((c + 0.5) * sq), int((r + 0.5) * sq))
        cv2.circle(img, center, int(sq * 0.3), PIECE, -1)
    return img


def occupied_squares(board) -> set[str]:
    import chess

    return {chess.square_name(sq) for sq in board.piece_map()}
```

`tests/test_move_detect.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_move_detect.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'chessbot.vision.move_detect'`

- [ ] **Step 3: Write the implementation**

`chessbot/vision/move_detect.py`:

```python
import chess
import cv2
import numpy as np

from .squares import get_square_img, row_col_to_square_name, square_to_row_col

EMPTY_STD = 15.0
CHANGE_MEAN = 10.0

_CASTLES = [
    ({"e8", "h8"}, {"f8", "g8"}, "e8g8"),
    ({"e8", "a8"}, {"c8", "d8"}, "e8c8"),
    ({"e1", "h1"}, {"f1", "g1"}, "e1g1"),
    ({"e1", "a1"}, {"c1", "d1"}, "e1c1"),
]


def is_empty(square_img: np.ndarray) -> bool:
    return float(square_img.std()) < EMPTY_STD


def board_changed(old: np.ndarray, new: np.ndarray) -> bool:
    for r in range(8):
        for c in range(8):
            if cv2.absdiff(get_square_img(r, c, old), get_square_img(r, c, new)).mean() > CHANGE_MEAN:
                return True
    return False


def _diff_squares(old, new, white_at_bottom):
    starts, ends = [], []
    for r in range(8):
        for c in range(8):
            old_sq = get_square_img(r, c, old)
            new_sq = get_square_img(r, c, new)
            if cv2.absdiff(old_sq, new_sq).mean() <= CHANGE_MEAN:
                continue
            name = row_col_to_square_name(r, c, white_at_bottom)
            if is_empty(new_sq):
                if not is_empty(old_sq):
                    starts.append(name)
            else:
                ends.append(name)
    return starts, ends


def find_candidate_moves(old, new, white_at_bottom: bool, board: chess.Board) -> list[str]:
    starts, ends = _diff_squares(old, new, white_at_bottom)
    for start_req, end_req, uci in _CASTLES:
        if start_req <= set(starts) and end_req <= set(ends):
            return [uci]
    moves = [s + e for s in starts for e in ends]
    if len(moves) > 20:
        return []
    valid = []
    for move in moves:
        if chess.Move.from_uci(move + "q") in board.legal_moves:
            return [move + "q"]
        if chess.Move.from_uci(move) in board.legal_moves:
            valid.append(move)
    if len(valid) == 2 and board.move_stack:
        last_to = chess.square_name(board.peek().to_square)
        return [valid[0]] if valid[0][2:4] != last_to else [valid[1]]
    if len(valid) > 2:
        return []
    return valid


def detect_stealth_move(img, board: chess.Board, white_at_bottom: bool) -> str | None:
    """Position drifted without a frame diff being observed: compare board state to image."""
    start, end = None, None
    for sq in chess.SQUARES:
        r, c = square_to_row_col(sq, white_at_bottom)
        img_empty = is_empty(get_square_img(r, c, img))
        if board.piece_at(sq) is not None and img_empty:
            start = chess.square_name(sq)
        if board.piece_at(sq) is None and not img_empty:
            end = chess.square_name(sq)
    if start and end:
        move = start + end
        if chess.Move.from_uci(move) in board.legal_moves:
            return move
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_move_detect.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add chessbot/vision/move_detect.py tests/test_move_detect.py tests/helpers.py
git commit -m "refactor: port diff-based move detection without process exits"
```

---

### Task 6: Capture layer — `chessbot/capture/`

**Files:**
- Create: `chessbot/capture/base.py`, `chessbot/capture/mss_backend.py`, `chessbot/capture/dxcam_backend.py`
- Modify: `chessbot/capture/__init__.py`
- Test: `tests/test_capture.py`

**Interfaces:**
- Produces:
  - `Capturer` protocol: `.grab(region: Region | None = None) -> np.ndarray` (BGR uint8; `region` in capture pixels), `.scale -> float` (capture pixels per screen point; mouse code divides pixel coords by this), `.name -> str`, `.close() -> None`
  - `get_capturer(backend: str = "auto") -> Capturer` — `"auto"`: dxcam on Windows with mss fallback, mss elsewhere.

Key subtlety (macOS Retina): mss grab boxes are specified in screen *points* but returned images are in physical *pixels* (2× on Retina). The backend therefore computes `scale` from an actual full grab vs the monitor dict, converts pixel regions to point boxes internally, and everything outside the capture layer works purely in capture pixels.

dxcam subtlety: `grab()` returns `None` when the screen hasn't changed since the last frame. The backend caches the last full frame, always grabs full-screen, and slices regions from it.

- [ ] **Step 1: Write the failing test**

`tests/test_capture.py`:

```python
import sys

import numpy as np
import pytest

from chessbot.capture import get_capturer


def _capturer_or_skip():
    try:
        cap = get_capturer("mss")
        img = cap.grab()
    except Exception as exc:  # no display / no screen-recording permission
        pytest.skip(f"screen capture unavailable: {exc}")
    return cap, img


def test_mss_full_grab_shape():
    cap, img = _capturer_or_skip()
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[0] > 100 and img.shape[1] > 100
    cap.close()


def test_mss_region_grab():
    cap, full = _capturer_or_skip()
    region = (0, 0, 128, 128)
    img = cap.grab(region)
    assert img.shape[2] == 3
    assert abs(img.shape[0] - 128) <= 4 and abs(img.shape[1] - 128) <= 4
    cap.close()


def test_mss_scale_positive():
    cap, _ = _capturer_or_skip()
    assert cap.scale >= 1.0
    cap.close()


def test_factory_auto_selects_platform_backend():
    cap, _ = _capturer_or_skip()
    cap.close()
    auto = get_capturer("auto")
    if sys.platform == "win32":
        assert auto.name in ("dxcam", "mss")
    else:
        assert auto.name == "mss"
    auto.close()


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError):
        get_capturer("nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_capture.py -q`
Expected: FAIL — `ImportError: cannot import name 'get_capturer'`

- [ ] **Step 3: Write the implementation**

`chessbot/capture/base.py`:

```python
from typing import Optional, Protocol, Tuple

import numpy as np

Region = Tuple[int, int, int, int]


class Capturer(Protocol):
    name: str

    @property
    def scale(self) -> float: ...

    def grab(self, region: Optional[Region] = None) -> np.ndarray: ...

    def close(self) -> None: ...
```

`chessbot/capture/mss_backend.py`:

```python
import mss
import numpy as np

from .base import Region


class MssCapturer:
    name = "mss"

    def __init__(self, monitor_index: int = 1):
        self._sct = mss.mss()
        self._monitor = self._sct.monitors[monitor_index]
        self._scale: float | None = None

    @property
    def scale(self) -> float:
        if self._scale is None:
            full = self._sct.grab(self._monitor)
            self._scale = full.width / self._monitor["width"]
        return self._scale

    def grab(self, region: Region | None = None) -> np.ndarray:
        if region is None:
            box = self._monitor
        else:
            s = self.scale
            left, top, right, bottom = region
            box = {
                "left": self._monitor["left"] + round(left / s),
                "top": self._monitor["top"] + round(top / s),
                "width": max(1, round((right - left) / s)),
                "height": max(1, round((bottom - top) / s)),
            }
        shot = self._sct.grab(box)
        return np.ascontiguousarray(np.asarray(shot)[:, :, :3])

    def close(self) -> None:
        self._sct.close()
```

`chessbot/capture/dxcam_backend.py`:

```python
import time

import numpy as np

from .base import Region


class DxcamCapturer:
    """Windows Desktop Duplication capture. dxcam returns BGR when configured so,
    and returns None from grab() when nothing changed; we cache the last frame."""

    name = "dxcam"

    def __init__(self):
        import dxcam

        self._cam = dxcam.create(output_color="BGR")
        if self._cam is None:
            raise RuntimeError("dxcam could not create a camera")
        self._last: np.ndarray | None = None

    @property
    def scale(self) -> float:
        return 1.0

    def grab(self, region: Region | None = None) -> np.ndarray:
        frame = self._cam.grab()
        if frame is not None:
            self._last = frame
        elif self._last is None:
            deadline = time.time() + 2.0
            while self._last is None and time.time() < deadline:
                frame = self._cam.grab()
                if frame is not None:
                    self._last = frame
                else:
                    time.sleep(0.005)
            if self._last is None:
                raise RuntimeError("dxcam produced no frames")
        img = self._last
        if region is not None:
            left, top, right, bottom = region
            img = img[top:bottom, left:right]
        return np.ascontiguousarray(img)

    def close(self) -> None:
        self._cam.release()
```

`chessbot/capture/__init__.py`:

```python
import sys

from .base import Capturer, Region
from .mss_backend import MssCapturer


def get_capturer(backend: str = "auto") -> Capturer:
    if backend == "mss":
        return MssCapturer()
    if backend == "dxcam":
        from .dxcam_backend import DxcamCapturer

        return DxcamCapturer()
    if backend == "auto":
        if sys.platform == "win32":
            try:
                from .dxcam_backend import DxcamCapturer

                return DxcamCapturer()
            except Exception:
                pass
        return MssCapturer()
    raise ValueError(f"unknown capture backend: {backend!r}")


__all__ = ["Capturer", "Region", "get_capturer", "MssCapturer"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_capture.py -q`
Expected: 5 passed (or skipped if the terminal lacks screen-recording permission — grant it in System Settings › Privacy & Security › Screen Recording and re-run).

- [ ] **Step 5: Commit**

```bash
git add chessbot/capture tests/test_capture.py
git commit -m "feat: cross-platform capture layer with mss and dxcam backends"
```

---

### Task 7: Config loading — `chessbot/config.py`

**Files:**
- Create: `chessbot/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `REPO_ROOT: Path` (parent of the `chessbot` package — valid for editable installs, which is the supported layout)
  - `@dataclass Settings: threads=4, hash_mb=512, depth=12, move_time=2.0, book_path=REPO_ROOT/"assets/Performance.bin", model_path=REPO_ROOT/"models/piece_classifier.onnx"`
  - `load_settings(ini_path: str | Path = "default.ini") -> Settings` (missing file/sections → defaults)

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:

```python
from chessbot.config import Settings, load_settings


def test_defaults_when_file_missing(tmp_path):
    settings = load_settings(tmp_path / "nope.ini")
    assert settings == Settings()


def test_loads_legacy_ini(tmp_path):
    ini = tmp_path / "default.ini"
    ini.write_text(
        "[Engine Default Settings]\nthread=2\nhash=256\n\n[Time Control]\ndepth=15\ntime=1.5\n"
    )
    settings = load_settings(ini)
    assert settings.threads == 2
    assert settings.hash_mb == 256
    assert settings.depth == 15
    assert settings.move_time == 1.5


def test_partial_ini_keeps_defaults(tmp_path):
    ini = tmp_path / "default.ini"
    ini.write_text("[Engine Default Settings]\nthread=8\n")
    settings = load_settings(ini)
    assert settings.threads == 8
    assert settings.hash_mb == Settings().hash_mb


def test_paths_exist_in_repo():
    assert Settings().book_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'chessbot.config'`

- [ ] **Step 3: Write the implementation**

`chessbot/config.py`:

```python
import configparser
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    threads: int = 4
    hash_mb: int = 512
    depth: int = 12
    move_time: float = 2.0
    book_path: Path = field(default_factory=lambda: REPO_ROOT / "assets" / "Performance.bin")
    model_path: Path = field(default_factory=lambda: REPO_ROOT / "models" / "piece_classifier.onnx")


def load_settings(ini_path: str | Path = "default.ini") -> Settings:
    settings = Settings()
    parser = configparser.ConfigParser()
    if not parser.read(ini_path):
        return settings
    engine = parser["Engine Default Settings"] if parser.has_section("Engine Default Settings") else {}
    tc = parser["Time Control"] if parser.has_section("Time Control") else {}
    settings.threads = int(engine.get("thread", settings.threads))
    settings.hash_mb = int(engine.get("hash", settings.hash_mb))
    settings.depth = int(tc.get("depth", settings.depth))
    settings.move_time = float(tc.get("time", settings.move_time))
    return settings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add chessbot/config.py tests/test_config.py
git commit -m "feat: settings loader for legacy default.ini"
```

---

### Task 8: Engine wrapper, book, time controls — `chessbot/engine/uci.py`

Ports `_parse_tc` from `bot-offline.py:814-875` and the engine/book handling from `play()`.

**Files:**
- Create: `chessbot/engine/uci.py`
- Test: `tests/test_uci.py`

**Interfaces:**
- Consumes: `Settings` fields (threads, hash_mb).
- Produces:
  - `parse_tc(tc: str) -> float` (per-move seconds; `ValueError` on malformed input)
  - `sample_timing(window: tuple[float, float], rng=random) -> float`
  - `class Book: __init__(path)`, `probe(board) -> str | None` (UCI or None)
  - `class EngineClient: __init__(path: str, threads: int, hash_mb: int, skill_level: int = 20)`, `best_move(board, *, depth: int | None = None, move_time: float | None = None) -> str`, `close()`
  - `find_engine(explicit: str | None) -> str` (explicit path or `shutil.which("stockfish")`; `FileNotFoundError` with a `--engine` hint otherwise)

- [ ] **Step 1: Write the failing test**

`tests/test_uci.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_uci.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'chessbot.engine.uci'`

- [ ] **Step 3: Write the implementation**

`chessbot/engine/uci.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_uci.py -q`
Expected: 8+ passed (engine test skips without stockfish; `brew install stockfish` to exercise it).

- [ ] **Step 5: Commit**

```bash
git add chessbot/engine/uci.py tests/test_uci.py
git commit -m "feat: engine client, opening book, and time control parsing"
```

---

### Task 9: Recognizer inference — `chessbot/vision/recognizer.py`

**Files:**
- Create: `chessbot/vision/recognizer.py`
- Test: `tests/test_recognizer.py`

**Interfaces:**
- Consumes: `CLASSES` from Task 3.
- Produces:
  - `INPUT_SIZE = 48`
  - `split_board(board_img: np.ndarray, size: int = 48) -> np.ndarray` shape `(64, size, size, 3)`, square index = `row * 8 + col`
  - `softmax(logits: np.ndarray) -> np.ndarray` (row-wise)
  - `class Recognizer: __init__(model_path=None, net=None)` (`net` injectable for tests; exactly one of the two), `classify_squares(board_img) -> tuple[list[list[str]], np.ndarray]` — 8×8 label grid + 8×8 float32 confidence array
- Model contract (from Global Constraints): blob = `cv2.dnn.blobFromImages(crops, 1/255.0, (48, 48), swapRB=False)`, output logits `(64, 13)`.

- [ ] **Step 1: Write the failing test**

`tests/test_recognizer.py`:

```python
import numpy as np
import pytest

from chessbot.vision.position import CLASSES
from chessbot.vision.recognizer import INPUT_SIZE, Recognizer, softmax, split_board


class FakeNet:
    def __init__(self, logits):
        self.logits = logits
        self.last_input = None

    def setInput(self, blob):
        self.last_input = blob

    def forward(self):
        return self.logits


def test_split_board_shape_and_order():
    img = np.zeros((400, 400, 3), np.uint8)
    img[0:50, 0:50] = 255
    crops = split_board(img, INPUT_SIZE)
    assert crops.shape == (64, INPUT_SIZE, INPUT_SIZE, 3)
    assert crops[0].mean() > 200
    assert crops[63].mean() < 10


def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = softmax(logits)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs[0].argmax() == 2


def test_recognizer_maps_logits_to_grid():
    logits = np.full((64, 13), -5.0, np.float32)
    logits[:, 0] = 5.0
    logits[0, 0] = -5.0
    logits[0, CLASSES.index("bR")] = 5.0
    logits[63, 0] = -5.0
    logits[63, CLASSES.index("wK")] = 5.0
    rec = Recognizer(net=FakeNet(logits))
    grid, conf = rec.classify_squares(np.zeros((400, 400, 3), np.uint8))
    assert grid[0][0] == "bR"
    assert grid[7][7] == "wK"
    assert grid[3][3] == "empty"
    assert conf.shape == (8, 8)
    assert conf.min() > 0.99


def test_recognizer_blob_contract():
    logits = np.zeros((64, 13), np.float32)
    net = FakeNet(logits)
    rec = Recognizer(net=net)
    img = np.full((160, 160, 3), 255, np.uint8)
    rec.classify_squares(img)
    blob = net.last_input
    assert blob.shape == (64, 3, INPUT_SIZE, INPUT_SIZE)
    assert blob.max() <= 1.0 and blob.max() > 0.99


def test_recognizer_requires_model_or_net():
    with pytest.raises(ValueError):
        Recognizer()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_recognizer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'chessbot.vision.recognizer'`

- [ ] **Step 3: Write the implementation**

`chessbot/vision/recognizer.py`:

```python
from pathlib import Path

import cv2
import numpy as np

from .position import CLASSES

INPUT_SIZE = 48


def split_board(board_img: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    h, w = board_img.shape[:2]
    crops = np.empty((64, size, size, 3), np.uint8)
    for row in range(8):
        y0, y1 = int(row * h / 8), int((row + 1) * h / 8)
        for col in range(8):
            x0, x1 = int(col * w / 8), int((col + 1) * w / 8)
            crops[row * 8 + col] = cv2.resize(board_img[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
    return crops


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class Recognizer:
    def __init__(self, model_path: str | Path | None = None, net=None):
        if net is None and model_path is None:
            raise ValueError("model_path or net is required")
        self._net = net if net is not None else cv2.dnn.readNetFromONNX(str(model_path))

    def classify_squares(self, board_img: np.ndarray) -> tuple[list[list[str]], np.ndarray]:
        crops = split_board(board_img)
        blob = cv2.dnn.blobFromImages(crops, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=False)
        self._net.setInput(blob)
        probs = softmax(np.asarray(self._net.forward()))
        indices = probs.argmax(axis=1)
        confidences = probs.max(axis=1).astype(np.float32).reshape(8, 8)
        grid = [[CLASSES[indices[row * 8 + col]] for col in range(8)] for row in range(8)]
        return grid, confidences
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_recognizer.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add chessbot/vision/recognizer.py tests/test_recognizer.py
git commit -m "feat: ONNX square classifier inference via cv2.dnn"
```

---

### Task 10: Mouse control — `chessbot/control/mouse.py`

**Files:**
- Create: `chessbot/control/mouse.py`
- Test: `tests/test_mouse.py`

**Interfaces:**
- Consumes: `square_center` from Task 2; `Capturer.scale` semantics from Task 6.
- Produces:
  - `class Mouse: __init__(scale: float, gui=None)` (`gui` defaults to pyautogui, injectable), `click_pixel(x, y)`, `click_square(name, white_at_bottom, region)`, `play_move(uci, white_at_bottom, region)` — clicks start then end square; for a 5-char promotion UCI also clicks the end square once more after `promotion_delay` (auto-queen: on lichess/chess.com the queen option renders on the promotion square itself).

- [ ] **Step 1: Write the failing test**

`tests/test_mouse.py`:

```python
from chessbot.control.mouse import Mouse


class FakeGui:
    def __init__(self):
        self.clicks = []

    def click(self, x, y):
        self.clicks.append((x, y))


def test_click_pixel_applies_scale():
    gui = FakeGui()
    mouse = Mouse(scale=2.0, gui=gui)
    mouse.click_pixel(100, 300)
    assert gui.clicks == [(50.0, 150.0)]


def test_play_move_clicks_centers():
    gui = FakeGui()
    mouse = Mouse(scale=1.0, gui=gui)
    mouse.play_move("e2e4", True, (0, 0, 800, 800))
    assert gui.clicks == [(450.0, 650.0), (450.0, 450.0)]


def test_play_move_promotion_clicks_end_square_again():
    gui = FakeGui()
    mouse = Mouse(scale=1.0, gui=gui, promotion_delay=0.0)
    mouse.play_move("e7e8q", True, (0, 0, 800, 800))
    assert len(gui.clicks) == 3
    assert gui.clicks[1] == gui.clicks[2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mouse.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'chessbot.control.mouse'`

- [ ] **Step 3: Write the implementation**

`chessbot/control/mouse.py`:

```python
import time

from ..vision.squares import square_center


class Mouse:
    def __init__(self, scale: float, gui=None, promotion_delay: float = 0.2):
        if gui is None:
            import pyautogui

            pyautogui.PAUSE = 0.01
            gui = pyautogui
        self._gui = gui
        self._scale = scale
        self._promotion_delay = promotion_delay

    def click_pixel(self, x: float, y: float) -> None:
        self._gui.click(x / self._scale, y / self._scale)

    def click_square(self, name: str, white_at_bottom: bool, region: tuple) -> None:
        x, y = square_center(name, white_at_bottom, region)
        self.click_pixel(x, y)

    def play_move(self, uci: str, white_at_bottom: bool, region: tuple) -> None:
        self.click_square(uci[:2], white_at_bottom, region)
        self.click_square(uci[2:4], white_at_bottom, region)
        if len(uci) == 5:
            time.sleep(self._promotion_delay)
            self.click_square(uci[2:4], white_at_bottom, region)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mouse.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add chessbot/control/mouse.py tests/test_mouse.py
git commit -m "feat: scaled mouse control with auto-queen promotion"
```

---

### Task 11: Game session — `chessbot/game.py`

Port of the main loop from `bot-offline.py:592-797` on top of the new components. Behavior changes: initial position comes from the recognizer (any position, not just startpos); every dead-end that previously called `os._exit(1)` either retries or re-syncs from the recognizer; timing modes (`delay`/`engine`/`both`), book usage (first 5 moves), and the depth-deepening-on-captures heuristic are preserved.

**Files:**
- Create: `chessbot/game.py`, `tests/fakes.py`
- Test: `tests/test_game.py`

**Interfaces:**
- Consumes: `locate_board` (T4), `move_detect` (T5), `Capturer` (T6), `Book`/`EngineClient`/`sample_timing` (T8), `Recognizer` (T9), `Mouse` (T10), `position`/`squares` (T2/T3).
- Produces:
  - `class GameSession(capturer, engine, book, recognizer, mouse, *, depth_mode=False, depth=12, move_time=2.0, timing_mode=None, timing_window=None, turn_arg=None, confidence_floor=0.90, confirm=input, log=print)`
  - `.locate(attempts=30, wait=1.0) -> Region` (sets `self.region`; raises `BoardNotFound` after retries)
  - `.read_position() -> tuple[chess.Board, bool]` (board + white_at_bottom)
  - `.run() -> None` (plays until game over)
  - `class BoardNotFound(RuntimeError)`

- [ ] **Step 1: Write fakes and failing tests**

`tests/fakes.py`:

```python
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
```

`tests/test_game.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_game.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'chessbot.game'`

- [ ] **Step 3: Write the implementation**

`chessbot/game.py`:

```python
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
                old = self._execute_move(move, old)
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

    def _execute_move(self, move: str, old):
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
            if moves:
                return ("move", moves[0], current)
            stable_misses += 1
            if stable_misses > 40:
                return ("resync", self._resync())

    def _resync(self) -> chess.Board:
        our_color = chess.WHITE if self.white_at_bottom else chess.BLACK
        img = self.capturer.grab(self.region)
        grid, _ = self.recognizer.classify_squares(img)
        return grid_to_board(grid, self.white_at_bottom, our_color)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_game.py -q`
Expected: 10 passed. Then run the whole suite: `pytest -q` — everything green.

- [ ] **Step 5: Commit**

```bash
git add chessbot/game.py tests/test_game.py tests/fakes.py
git commit -m "feat: game session with recognizer bootstrap and resync fallback"
```

---

### Task 12: Move server + remote engine — `chessbot/engine/server.py`, `chessbot/engine/remote.py`

Port of `bot_server.py` and the socket client from `bot-online.py:680-790`. Wire protocol preserved: client sends `"{fen},{tc},{limit}"` (≤128 bytes UTF-8), server replies with a UCI move string. The lichess tablebase probe for <8 pieces is kept behind a lazy `requests` import (part of the `[training]` extra; the server degrades gracefully without it).

**Files:**
- Create: `chessbot/engine/server.py`, `chessbot/engine/remote.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `EngineClient` (T8) duck-type: `best_move(board, *, depth=None, move_time=None) -> str`.
- Produces:
  - `server.compute_move(request: str, engine) -> str` (parses protocol, `ValueError` on garbage)
  - `server.serve(engine, host: str, port: int)` (blocking accept loop, thread per client)
  - `class remote.RemoteEngine: __init__(host, port)`, `best_move(board, *, depth=None, move_time=None) -> str`, `close()` — same duck-type as `EngineClient`, so `GameSession` accepts it unchanged.

- [ ] **Step 1: Write the failing test**

`tests/test_server.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'chessbot.engine.remote'`

- [ ] **Step 3: Write the implementation**

`chessbot/engine/server.py`:

```python
import socket
import threading

import chess

ENCODING = "utf-8"
TABLEBASE_URL = "http://tablebase.lichess.ovh/standard"


def _tablebase_move(fen: str) -> str | None:
    if len(chess.Board(fen).piece_map()) >= 8:
        return None
    try:
        import requests

        response = requests.get(TABLEBASE_URL, params={"fen": fen.replace(" ", "_")}, timeout=5)
        if response.status_code == 200:
            return response.json()["moves"][0]["uci"]
    except Exception:
        return None
    return None


def compute_move(request: str, engine) -> str:
    try:
        fen, tc, limit = (part.strip() for part in request.split(","))
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"malformed request: {request!r}") from exc
    move = _tablebase_move(fen)
    if move is not None:
        return move
    if tc == "depth":
        return engine.best_move(board, depth=int(limit))
    return engine.best_move(board, move_time=float(limit))


def _handle_client(conn: socket.socket, engine) -> None:
    with conn:
        while True:
            try:
                data = conn.recv(128)
            except ConnectionResetError:
                return
            if not data:
                return
            try:
                move = compute_move(data.decode(ENCODING), engine)
            except ValueError as exc:
                print(f"Dropping client: {exc}")
                return
            conn.sendall(move.encode(ENCODING))
            print(f"Sent move: {move}")


def serve(engine, host: str = "0.0.0.0", port: int = 6751) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen()
        print(f"Serving moves on {host}:{port}")
        while True:
            conn, addr = server.accept()
            print(f"Connected: {addr}")
            threading.Thread(target=_handle_client, args=(conn, engine), daemon=True).start()
```

`chessbot/engine/remote.py`:

```python
import socket

import chess

from .server import ENCODING


class RemoteEngine:
    def __init__(self, host: str, port: int):
        self._sock = socket.create_connection((host, port), timeout=60)

    def best_move(self, board: chess.Board, *, depth: int | None = None, move_time: float | None = None) -> str:
        if depth is not None:
            request = f"{board.fen()},depth,{depth}"
        else:
            request = f"{board.fen()},time,{move_time if move_time else 0.1}"
        self._sock.sendall(request.encode(ENCODING))
        return self._sock.recv(128).decode(ENCODING)

    def close(self) -> None:
        self._sock.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_server.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add chessbot/engine/server.py chessbot/engine/remote.py tests/test_server.py
git commit -m "refactor: port move server and remote engine client"
```

---

### Task 13: CLI — `chessbot/cli.py`

**Files:**
- Create: `chessbot/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything runtime — `load_settings` (T7), `get_capturer` (T6), `find_engine`/`EngineClient`/`Book`/`parse_tc` (T8), `RemoteEngine` (T12), `Recognizer` (T9), `Mouse` (T10), `GameSession` (T11), `serve` (T12).
- Produces:
  - `build_parser() -> argparse.ArgumentParser` with `play` and `server` subcommands (flags per spec: `--engine --turn --capture --config --model --tc --depth-mode --depth --threads --hash --timing-mode --timing-min --timing-max --remote`; server: `--port --engine --threads --hash`)
  - `resolve_timing(args) -> tuple[str | None, tuple[float, float] | None]` (raises `SystemExit` with message on invalid combos)
  - `main(argv=None) -> None` — console entry point.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:

```python
import pytest

from chessbot.cli import build_parser, resolve_timing


def test_play_defaults():
    args = build_parser().parse_args(["play"])
    assert args.command == "play"
    assert args.capture == "auto"
    assert args.turn is None


def test_play_all_flags():
    args = build_parser().parse_args(
        [
            "play", "--engine", "/usr/bin/stockfish", "--turn", "black", "--capture", "mss",
            "--tc", "40/5m", "--depth-mode", "--depth", "14", "--threads", "2", "--hash", "256",
            "--timing-mode", "both", "--timing-min", "1", "--timing-max", "3",
            "--remote", "1.2.3.4:6751", "--model", "models/piece_classifier.onnx",
        ]
    )
    assert args.turn == "black" and args.depth == 14 and args.remote == "1.2.3.4:6751"


def test_server_flags():
    args = build_parser().parse_args(["server", "--port", "7000", "--hash", "2048"])
    assert args.command == "server" and args.port == 7000


def test_resolve_timing_valid():
    args = build_parser().parse_args(
        ["play", "--timing-mode", "delay", "--timing-min", "1", "--timing-max", "2"]
    )
    assert resolve_timing(args) == ("delay", (1.0, 2.0))


def test_resolve_timing_none():
    args = build_parser().parse_args(["play"])
    assert resolve_timing(args) == (None, None)


@pytest.mark.parametrize(
    "flags",
    [
        ["--timing-mode", "delay"],
        ["--timing-mode", "delay", "--timing-min", "2", "--timing-max", "1"],
        ["--timing-mode", "delay", "--timing-min", "-1", "--timing-max", "1"],
    ],
)
def test_resolve_timing_invalid(flags):
    args = build_parser().parse_args(["play", *flags])
    with pytest.raises(SystemExit):
        resolve_timing(args)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'chessbot.cli'`

- [ ] **Step 3: Write the implementation**

`chessbot/cli.py`:

```python
import argparse
import sys

import chess

from .capture import get_capturer
from .config import Settings, load_settings
from .engine.uci import Book, EngineClient, find_engine, parse_tc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chessbot", description="Screen-reading chess bot")
    sub = parser.add_subparsers(dest="command", required=True)

    play = sub.add_parser("play", help="detect the on-screen board and play")
    play.add_argument("--engine", help="path to a UCI engine (default: stockfish on PATH)")
    play.add_argument("--turn", choices=["white", "black"], help="side to move in the detected position")
    play.add_argument("--capture", choices=["auto", "mss", "dxcam"], default="auto")
    play.add_argument("--config", default="default.ini")
    play.add_argument("--model", help="path to the piece classifier ONNX model")
    play.add_argument("--tc", help='classical control "40/Xu" (u=ms|s|m|h, default minutes)')
    play.add_argument("--depth-mode", action="store_true")
    play.add_argument("--depth", type=int)
    play.add_argument("--threads", type=int)
    play.add_argument("--hash", type=int)
    play.add_argument("--timing-mode", choices=["delay", "engine", "both"])
    play.add_argument("--timing-min", type=float)
    play.add_argument("--timing-max", type=float)
    play.add_argument("--remote", help="host:port of a chessbot server to use instead of a local engine")

    server = sub.add_parser("server", help="serve engine moves over a socket")
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=6751)
    server.add_argument("--engine")
    server.add_argument("--threads", type=int)
    server.add_argument("--hash", type=int, default=1024)
    return parser


def resolve_timing(args) -> tuple[str | None, tuple[float, float] | None]:
    if not args.timing_mode:
        return None, None
    if args.timing_min is None or args.timing_max is None:
        raise SystemExit("--timing-mode requires both --timing-min and --timing-max")
    if args.timing_min < 0 or args.timing_max < 0:
        raise SystemExit("timing bounds must be non-negative")
    if args.timing_min > args.timing_max:
        raise SystemExit("--timing-min must be <= --timing-max")
    return args.timing_mode, (float(args.timing_min), float(args.timing_max))


def _apply_overrides(settings: Settings, args) -> Settings:
    if args.threads is not None:
        settings.threads = max(1, args.threads)
    if args.hash is not None:
        settings.hash_mb = max(16, args.hash)
    if getattr(args, "depth", None) is not None:
        settings.depth = max(1, args.depth)
    return settings


def cmd_play(args) -> None:
    from .control.mouse import Mouse
    from .game import GameSession
    from .vision.recognizer import Recognizer

    settings = _apply_overrides(load_settings(args.config), args)
    if args.tc:
        settings.move_time = parse_tc(args.tc)
        print(f"Classical control: {settings.move_time:.3f}s per move")
    timing_mode, timing_window = resolve_timing(args)

    model_path = args.model or settings.model_path
    capturer = get_capturer(args.capture)
    if args.remote:
        from .engine.remote import RemoteEngine

        host, _, port = args.remote.partition(":")
        engine = RemoteEngine(host, int(port or 6751))
    else:
        engine = EngineClient(find_engine(args.engine), settings.threads, settings.hash_mb)
    turn_arg = None
    if args.turn:
        turn_arg = chess.WHITE if args.turn == "white" else chess.BLACK

    session = GameSession(
        capturer=capturer,
        engine=engine,
        book=Book(settings.book_path),
        recognizer=Recognizer(model_path=model_path),
        mouse=Mouse(scale=capturer.scale),
        depth_mode=args.depth_mode,
        depth=settings.depth,
        move_time=settings.move_time,
        timing_mode=timing_mode,
        timing_window=timing_window,
        turn_arg=turn_arg,
    )
    try:
        session.run()
    finally:
        engine.close()
        capturer.close()


def cmd_server(args) -> None:
    from .engine.server import serve

    threads = args.threads or max(1, (__import__("os").cpu_count() or 2) - 1)
    engine = EngineClient(find_engine(args.engine), threads, args.hash)
    try:
        serve(engine, args.host, args.port)
    finally:
        engine.close()


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "play":
            cmd_play(args)
        else:
            cmd_server(args)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))
    except KeyboardInterrupt:
        sys.exit("\ninterrupted")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -q && chessbot --help >/dev/null && chessbot play --help >/dev/null`
Expected: 8 passed; both help invocations exit 0.

- [ ] **Step 5: Commit**

```bash
git add chessbot/cli.py tests/test_cli.py
git commit -m "feat: chessbot CLI with play and server commands"
```

---

### Task 14: Asset fetcher — `training/fetch_assets.py`

Downloads lichess piece SVGs + board textures (lichess-org/lila, permissively licensed) and chess.com standard piece PNGs. Downloads land in `training/assets/` which is gitignored — chess.com assets are proprietary and must never be committed.

**Files:**
- Create: `training/__init__.py` (empty), `training/fetch_assets.py`
- Test: `tests/test_fetch_assets.py` (URL builders only, no network)

**Interfaces:**
- Produces:
  - `LICHESS_PIECE_SETS: list[str]`, `CHESSCOM_PIECE_SETS: list[str]`, `PIECE_CODES = ["wP",...,"bK"]` (12 entries)
  - `lichess_piece_url(piece_set: str, piece: str) -> str`
  - `chesscom_piece_url(piece_set: str, piece: str) -> str`
  - `lichess_board_url(filename: str) -> str`
  - `fetch_all(dest: Path) -> dict` (counts; skips existing files, warns and continues on HTTP errors)
  - CLI: `python -m training.fetch_assets [--dest training/assets]`
- Directory layout produced: `<dest>/lichess/<set>/wP.svg`, `<dest>/chesscom/<set>/wp.png`, `<dest>/boards/<filename>`

- [ ] **Step 1: Write the failing test**

`tests/test_fetch_assets.py`:

```python
from training.fetch_assets import (
    CHESSCOM_PIECE_SETS,
    LICHESS_PIECE_SETS,
    PIECE_CODES,
    chesscom_piece_url,
    lichess_board_url,
    lichess_piece_url,
)


def test_piece_codes():
    assert len(PIECE_CODES) == 12
    assert PIECE_CODES[0] == "wP" and PIECE_CODES[-1] == "bK"


def test_set_lists_nonempty():
    assert len(LICHESS_PIECE_SETS) >= 25
    assert "cburnett" in LICHESS_PIECE_SETS
    assert len(CHESSCOM_PIECE_SETS) >= 5
    assert "neo" in CHESSCOM_PIECE_SETS


def test_lichess_piece_url():
    assert (
        lichess_piece_url("cburnett", "wP")
        == "https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett/wP.svg"
    )


def test_chesscom_piece_url_lowercases():
    assert (
        chesscom_piece_url("neo", "bQ")
        == "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bq.png"
    )


def test_lichess_board_url():
    assert (
        lichess_board_url("wood.jpg")
        == "https://raw.githubusercontent.com/lichess-org/lila/master/public/images/board/wood.jpg"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fetch_assets.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.fetch_assets'`

- [ ] **Step 3: Write the implementation**

Create empty `training/__init__.py`, then `training/fetch_assets.py`:

```python
import argparse
from pathlib import Path

LILA_RAW = "https://raw.githubusercontent.com/lichess-org/lila/master/public"

LICHESS_PIECE_SETS = [
    "alpha", "anarcandy", "caliente", "california", "cardinal", "cburnett", "celtic",
    "chess7", "chessnut", "companion", "cooke", "dubrovny", "fantasy", "fresca",
    "gioco", "governor", "horsey", "icpieces", "kiwen-suwi", "kosal", "leipzig",
    "maestro", "merida", "monarchy", "mpchess", "pirouetti", "pixel", "reillycraig",
    "rhosgfx", "riohacha", "spatial", "staunty", "tatiana", "xkcd",
]

CHESSCOM_PIECE_SETS = ["neo", "classic", "wood", "glass", "gothic", "metal", "bases", "icy_sea"]

LICHESS_BOARD_IMAGES = [
    "wood.jpg", "wood2.jpg", "wood3.jpg", "wood4.jpg", "maple.jpg", "maple2.jpg",
    "marble.jpg", "grey.jpg", "metal.jpg", "olive.jpg", "blue-marble.jpg",
    "green-plastic.png", "leather.jpg", "canvas2.jpg", "horsey.jpg",
    "pink-pyramid.png", "purple-diag.png",
]

PIECE_CODES = ["wP", "wN", "wB", "wR", "wQ", "wK", "bP", "bN", "bB", "bR", "bQ", "bK"]


def lichess_piece_url(piece_set: str, piece: str) -> str:
    return f"{LILA_RAW}/piece/{piece_set}/{piece}.svg"


def chesscom_piece_url(piece_set: str, piece: str) -> str:
    return f"https://images.chesscomfiles.com/chess-themes/pieces/{piece_set}/150/{piece.lower()}.png"


def lichess_board_url(filename: str) -> str:
    return f"{LILA_RAW}/images/board/{filename}"


def _download(session, url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    response = session.get(url, timeout=30)
    if response.status_code != 200:
        print(f"skip ({response.status_code}): {url}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    return True


def fetch_all(dest: Path) -> dict:
    import requests

    session = requests.Session()
    counts = {"lichess": 0, "chesscom": 0, "boards": 0, "failed": 0}
    for piece_set in LICHESS_PIECE_SETS:
        for piece in PIECE_CODES:
            path = dest / "lichess" / piece_set / f"{piece}.svg"
            if _download(session, lichess_piece_url(piece_set, piece), path):
                counts["lichess"] += 1
            else:
                counts["failed"] += 1
    for piece_set in CHESSCOM_PIECE_SETS:
        for piece in PIECE_CODES:
            path = dest / "chesscom" / piece_set / f"{piece.lower()}.png"
            if _download(session, chesscom_piece_url(piece_set, piece), path):
                counts["chesscom"] += 1
            else:
                counts["failed"] += 1
    for filename in LICHESS_BOARD_IMAGES:
        if _download(session, lichess_board_url(filename), dest / "boards" / filename):
            counts["boards"] += 1
        else:
            counts["failed"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Download piece and board assets for training")
    parser.add_argument("--dest", type=Path, default=Path("training/assets"))
    args = parser.parse_args()
    counts = fetch_all(args.dest)
    print(counts)
    incomplete = [
        d.name
        for d in (args.dest / "lichess").iterdir()
        if d.is_dir() and len(list(d.glob("*.svg"))) != 12
    ]
    if incomplete:
        print(f"warning: incomplete lichess sets (will be skipped in training): {incomplete}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fetch_assets.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add training/__init__.py training/fetch_assets.py tests/test_fetch_assets.py
git commit -m "feat: training asset fetcher for lichess and chess.com piece sets"
```

---

### Task 15: Dataset generator — `training/generate_dataset.py`

Synthesizes labeled 48×48 square crops: piece image (SVG rendered via cairosvg, or PNG) composited over square backgrounds (flat theme colors or crops of real board textures) with augmentation. Also provides `render_full_board` used by the evaluator (Task 17) and fixture generation (Task 18).

**Files:**
- Create: `training/generate_dataset.py`
- Test: `tests/test_generate_dataset.py` (uses PIL-drawn fake piece sets — no network, no cairosvg)

**Interfaces:**
- Consumes: `CLASSES`, `board_to_grid` from Task 3; `PIECE_CODES` from Task 14; asset layout from Task 14.
- Produces:
  - `FLAT_THEMES: list[tuple[rgb, rgb]]`, `HIGHLIGHTS: list[rgba]`, `VAL_SETS: set[str]` = `{"staunty", "governor", "icpieces", "kosal", "glass"}`
  - `load_piece_set(set_dir: Path, render_px: int = 128) -> dict[str, PIL.Image]` (keys = PIECE_CODES; SVG or PNG autodetected; raises `ValueError` if any of the 12 is missing)
  - `render_square(piece_img, light_bg, dark_bg, is_light: bool, rng, out_size=48, augment=True) -> np.ndarray` (BGR 48×48)
  - `render_full_board(pieces: dict, board: chess.Board, theme: tuple, square_px=64, white_at_bottom=True) -> np.ndarray` (BGR, no augmentation — clean renders)
  - `build_dataset(assets_dir: Path, out_dir: Path, per_class: int, seed: int) -> None` → `out_dir/{train,val}/<class>/<set>_<i>.png`
  - CLI: `python -m training.generate_dataset [--assets training/assets] [--out training/dataset] [--per-class 160] [--seed 0]`

- [ ] **Step 1: Write the failing test**

`tests/test_generate_dataset.py`:

```python
import random

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw

from chessbot.vision.position import CLASSES
from training.fetch_assets import PIECE_CODES
from training.generate_dataset import (
    FLAT_THEMES,
    build_dataset,
    load_piece_set,
    render_full_board,
    render_square,
)


@pytest.fixture
def fake_set_dir(tmp_path):
    set_dir = tmp_path / "assets" / "chesscom" / "fake"
    set_dir.mkdir(parents=True)
    for i, code in enumerate(PIECE_CODES):
        img = Image.new("RGBA", (150, 150), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        color = (255, 255, 255, 255) if code.startswith("w") else (30, 30, 30, 255)
        draw.ellipse([20 + i, 20, 130 - i, 130], fill=color)
        img.save(set_dir / f"{code.lower()}.png")
    return tmp_path / "assets"


def test_load_piece_set(fake_set_dir):
    pieces = load_piece_set(fake_set_dir / "chesscom" / "fake")
    assert set(pieces) == set(PIECE_CODES)
    assert pieces["wP"].mode == "RGBA"


def test_load_piece_set_rejects_incomplete(tmp_path):
    empty = tmp_path / "incomplete"
    empty.mkdir()
    with pytest.raises(ValueError):
        load_piece_set(empty)


def test_render_square_shapes(fake_set_dir):
    pieces = load_piece_set(fake_set_dir / "chesscom" / "fake")
    rng = random.Random(0)
    light, dark = FLAT_THEMES[0]
    occupied = render_square(pieces["wK"], light, dark, True, rng)
    empty = render_square(None, light, dark, False, rng)
    assert occupied.shape == (48, 48, 3) and occupied.dtype == np.uint8
    assert empty.shape == (48, 48, 3)
    assert occupied.std() > empty.std()


def test_render_square_deterministic_with_seed(fake_set_dir):
    pieces = load_piece_set(fake_set_dir / "chesscom" / "fake")
    light, dark = FLAT_THEMES[0]
    a = render_square(pieces["bN"], light, dark, True, random.Random(42))
    b = render_square(pieces["bN"], light, dark, True, random.Random(42))
    assert np.array_equal(a, b)


def test_render_full_board(fake_set_dir):
    import chess

    pieces = load_piece_set(fake_set_dir / "chesscom" / "fake")
    img = render_full_board(pieces, chess.Board(), FLAT_THEMES[0], square_px=32)
    assert img.shape == (256, 256, 3)


def test_build_dataset_layout(fake_set_dir, tmp_path):
    out = tmp_path / "dataset"
    build_dataset(fake_set_dir, out, per_class=2, seed=0)
    for cls in CLASSES:
        files = list((out / "train" / cls).glob("*.png"))
        assert len(files) == 2, f"missing samples for {cls}"
    import cv2

    sample = cv2.imread(str(next((out / "train" / "wK").glob("*.png"))))
    assert sample.shape == (48, 48, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pip install -e ".[training]" -q && pytest tests/test_generate_dataset.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.generate_dataset'`
(The training extra installs torch/cairosvg/pillow/requests — needed from here on.)

- [ ] **Step 3: Write the implementation**

`training/generate_dataset.py`:

```python
import argparse
import io
import random
from pathlib import Path

import chess
import cv2
import numpy as np
from PIL import Image

from chessbot.vision.position import board_to_grid
from training.fetch_assets import PIECE_CODES

FLAT_THEMES = [
    ((240, 217, 181), (181, 136, 99)),
    ((222, 227, 230), (140, 162, 173)),
    ((255, 255, 221), (134, 166, 102)),
    ((238, 238, 210), (118, 150, 86)),
    ((234, 233, 210), (75, 115, 153)),
    ((240, 241, 240), (120, 120, 125)),
    ((235, 224, 206), (170, 138, 109)),
    ((255, 255, 255), (88, 120, 170)),
]

HIGHLIGHTS = [
    (155, 199, 0, 105),
    (255, 255, 51, 127),
    (0, 155, 199, 80),
    (255, 60, 60, 90),
]

VAL_SETS = {"staunty", "governor", "icpieces", "kosal", "glass"}


def load_piece_set(set_dir: Path, render_px: int = 128) -> dict[str, Image.Image]:
    pieces = {}
    for code in PIECE_CODES:
        svg = set_dir / f"{code}.svg"
        png = set_dir / f"{code.lower()}.png"
        if svg.exists():
            import cairosvg

            data = cairosvg.svg2png(url=str(svg), output_width=render_px, output_height=render_px)
            pieces[code] = Image.open(io.BytesIO(data)).convert("RGBA")
        elif png.exists():
            pieces[code] = Image.open(png).convert("RGBA")
        else:
            raise ValueError(f"piece set at {set_dir} is missing {code}")
    return pieces


def _load_texture_squares(boards_dir: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split each full-board texture into a (light, dark) pair of sample squares."""
    pairs = []
    if not boards_dir.is_dir():
        return pairs
    for path in sorted(boards_dir.iterdir()):
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, (512, 512))
        sq = 64
        light = img[0:sq, 0:sq]
        dark = img[0:sq, sq:2 * sq]
        pairs.append((cv2.cvtColor(light, cv2.COLOR_BGR2RGB), cv2.cvtColor(dark, cv2.COLOR_BGR2RGB)))
    return pairs


def _background(light_bg, dark_bg, is_light: bool, size: int) -> Image.Image:
    bg = light_bg if is_light else dark_bg
    if isinstance(bg, np.ndarray):
        return Image.fromarray(cv2.resize(bg, (size, size))).convert("RGBA")
    return Image.new("RGBA", (size, size), (*bg, 255))


def render_square(piece_img, light_bg, dark_bg, is_light: bool, rng: random.Random,
                  out_size: int = 48, augment: bool = True) -> np.ndarray:
    size = rng.randrange(56, 129) if augment else 96
    canvas = _background(light_bg, dark_bg, is_light, size)
    if augment and rng.random() < 0.30:
        overlay = Image.new("RGBA", (size, size), rng.choice(HIGHLIGHTS))
        canvas = Image.alpha_composite(canvas, overlay)
    if piece_img is not None:
        scale = rng.uniform(0.75, 0.97) if augment else 0.85
        piece_px = max(8, int(size * scale))
        piece = piece_img.resize((piece_px, piece_px), Image.LANCZOS)
        max_off = int(size * 0.06) if augment else 0
        ox = (size - piece_px) // 2 + (rng.randint(-max_off, max_off) if augment else 0)
        oy = (size - piece_px) // 2 + (rng.randint(-max_off, max_off) if augment else 0)
        canvas.alpha_composite(piece, (ox, oy))
    arr = cv2.cvtColor(np.asarray(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)
    if augment:
        alpha = rng.uniform(0.85, 1.15)
        beta = rng.uniform(-20, 20)
        arr = np.clip(arr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if rng.random() < 0.20:
            arr = cv2.GaussianBlur(arr, (3, 3), 0)
        if rng.random() < 0.35:
            quality = rng.randint(40, 90)
            _, enc = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            arr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.resize(arr, (out_size, out_size), interpolation=cv2.INTER_AREA)


def render_full_board(pieces: dict, board: chess.Board, theme, square_px: int = 64,
                      white_at_bottom: bool = True) -> np.ndarray:
    light, dark = theme
    size = square_px * 8
    canvas = Image.new("RGBA", (size, size))
    grid = board_to_grid(board, white_at_bottom)
    for row in range(8):
        for col in range(8):
            is_light = (row + col) % 2 == 0
            tile = _background(light, dark, is_light, square_px)
            label = grid[row][col]
            if label != "empty":
                piece_px = int(square_px * 0.85)
                piece = pieces[label].resize((piece_px, piece_px), Image.LANCZOS)
                pad = (square_px - piece_px) // 2
                tile.alpha_composite(piece, (pad, pad))
            canvas.paste(tile, (col * square_px, row * square_px))
    return cv2.cvtColor(np.asarray(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)


def _iter_piece_sets(assets_dir: Path):
    for source in ("lichess", "chesscom"):
        source_dir = assets_dir / source
        if not source_dir.is_dir():
            continue
        for set_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
            try:
                yield set_dir.name, load_piece_set(set_dir)
            except ValueError as exc:
                print(f"skipping {set_dir.name}: {exc}")


def build_dataset(assets_dir: Path, out_dir: Path, per_class: int = 160, seed: int = 0) -> None:
    rng = random.Random(seed)
    textures = _load_texture_squares(assets_dir / "boards")
    backgrounds = list(FLAT_THEMES) + textures
    sets = list(_iter_piece_sets(assets_dir))
    if not sets:
        raise SystemExit(f"no piece sets found under {assets_dir}; run fetch_assets first")
    for set_name, pieces in sets:
        split = "val" if set_name in VAL_SETS else "train"
        n = per_class if split == "train" else max(1, per_class // 3)
        for label in ["empty", *PIECE_CODES]:
            class_dir = out_dir / split / label
            class_dir.mkdir(parents=True, exist_ok=True)
            piece_img = None if label == "empty" else pieces[label]
            for i in range(n):
                light, dark = rng.choice(backgrounds)
                crop = render_square(piece_img, light, dark, rng.random() < 0.5, rng)
                cv2.imwrite(str(class_dir / f"{set_name}_{i}.png"), crop)
        print(f"{set_name} -> {split}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labeled square crops")
    parser.add_argument("--assets", type=Path, default=Path("training/assets"))
    parser.add_argument("--out", type=Path, default=Path("training/dataset"))
    parser.add_argument("--per-class", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build_dataset(args.assets, args.out, args.per_class, args.seed)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_generate_dataset.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add training/generate_dataset.py tests/test_generate_dataset.py
git commit -m "feat: synthetic square dataset generator with augmentation"
```

---

### Task 16: Trainer + ONNX export — `training/train.py`

**Files:**
- Create: `training/train.py`
- Test: `tests/test_training.py`

**Interfaces:**
- Consumes: `CLASSES` (T3), dataset layout from T15 (`<root>/{train,val}/<class>/*.png`).
- Produces:
  - `class PieceNet(torch.nn.Module)` — 3 conv blocks (24/48/96 ch, BN, ReLU, maxpool) + global average pool + linear to 13 logits, ~55K params
  - `class SquaresDataset(torch.utils.data.Dataset)` — scans class dirs, yields `(float32 CHW /255 BGR tensor, label_idx)`
  - `train_model(data_root: Path, epochs, batch_size, lr, device) -> tuple[PieceNet, float]` (returns best model + val accuracy)
  - `export_onnx(model, out_path: Path)` — opset 12, dynamic batch, input `squares`, output `logits`
  - CLI: `python -m training.train [--data training/dataset] [--out models/piece_classifier.onnx] [--epochs 12] [--batch 256] [--lr 1e-3] [--device auto]`

- [ ] **Step 1: Write the failing test**

`tests/test_training.py`:

```python
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL")

from PIL import Image, ImageDraw

from chessbot.vision.position import CLASSES
from training.train import PieceNet, SquaresDataset, export_onnx, train_model


@pytest.fixture
def tiny_dataset(tmp_path):
    rng = random.Random(0)
    for split, n in (("train", 4), ("val", 2)):
        for cls_idx, cls in enumerate(CLASSES):
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            for i in range(n):
                img = Image.new("RGB", (48, 48), (cls_idx * 19 % 255, 100, 150))
                draw = ImageDraw.Draw(img)
                draw.ellipse([10, 10, 38, 38], fill=(cls_idx * 7 % 255,) * 3)
                img.save(d / f"{i}.png")
    return tmp_path


def test_piecenet_output_shape():
    net = PieceNet()
    out = net(torch.zeros(2, 3, 48, 48))
    assert out.shape == (2, 13)


def test_piecenet_param_count():
    n = sum(p.numel() for p in PieceNet().parameters())
    assert n < 200_000


def test_dataset_loads(tiny_dataset):
    ds = SquaresDataset(tiny_dataset / "train")
    assert len(ds) == 4 * 13
    x, y = ds[0]
    assert x.shape == (3, 48, 48) and x.dtype == torch.float32
    assert x.max() <= 1.0
    assert 0 <= y < 13


def test_train_and_export_onnx_loads_in_cv2(tiny_dataset, tmp_path):
    import cv2

    model, _ = train_model(tiny_dataset, epochs=1, batch_size=8, lr=1e-3, device="cpu")
    out = tmp_path / "model.onnx"
    export_onnx(model, out)
    net = cv2.dnn.readNetFromONNX(str(out))
    blob = np.zeros((2, 3, 48, 48), np.float32)
    net.setInput(blob)
    logits = net.forward()
    assert logits.shape == (2, 13)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_training.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.train'`

- [ ] **Step 3: Write the implementation**

`training/train.py`:

```python
import argparse
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from chessbot.vision.position import CLASSES


class SquaresDataset(Dataset):
    def __init__(self, root: Path):
        self.samples = []
        for idx, cls in enumerate(CLASSES):
            for path in sorted((Path(root) / cls).glob("*.png")):
                self.samples.append((path, idx))
        if not self.samples:
            raise ValueError(f"no samples under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = cv2.imread(str(path))
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        return tensor, label


class PieceNet(nn.Module):
    def __init__(self, n_classes: int = 13):
        super().__init__()
        def block(cin, cout):
            return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2)]

        self.features = nn.Sequential(*block(3, 24), *block(24, 48), *block(48, 96), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(96, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def _accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(dim=1).cpu()
            correct += int((pred == y).sum())
            total += len(y)
    return correct / max(1, total)


def _pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_model(data_root: Path, epochs: int = 12, batch_size: int = 256, lr: float = 1e-3,
                device: str = "auto"):
    device = _pick_device(device)
    train_loader = DataLoader(SquaresDataset(Path(data_root) / "train"), batch_size=batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(SquaresDataset(Path(data_root) / "val"), batch_size=batch_size, num_workers=2)
    model = PieceNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
        acc = _accuracy(model, val_loader, device)
        print(f"epoch {epoch + 1}/{epochs}: val acc {acc:.4%}")
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.cpu(), best_acc


def export_onnx(model, out_path: Path) -> None:
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 48, 48),
        str(out_path),
        input_names=["squares"],
        output_names=["logits"],
        dynamic_axes={"squares": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=12,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the piece classifier and export ONNX")
    parser.add_argument("--data", type=Path, default=Path("training/dataset"))
    parser.add_argument("--out", type=Path, default=Path("models/piece_classifier.onnx"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    model, acc = train_model(args.data, args.epochs, args.batch, args.lr, args.device)
    export_onnx(model, args.out)
    print(f"exported {args.out} (best val acc {acc:.4%})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_training.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add training/train.py tests/test_training.py
git commit -m "feat: piece classifier training and ONNX export"
```

---

### Task 17: Evaluator — `training/evaluate.py`

**Files:**
- Create: `training/evaluate.py`
- Test: covered by Task 16's ONNX test plus a full-board unit test added here.

**Interfaces:**
- Consumes: `Recognizer` (T9), `render_full_board`/`load_piece_set`/`FLAT_THEMES`/`VAL_SETS` (T15), dataset layout (T15).
- Produces:
  - `eval_squares(model_path, data_root) -> dict` — `{"overall": float, "per_class": {label: acc}}` over `val/`
  - `eval_boards(model_path, assets_dir, n_positions=40, seed=0) -> dict` — renders random legal midgame positions with held-out piece sets, `{"per_square": float, "exact_boards": float}`
  - CLI: `python -m training.evaluate [--model models/piece_classifier.onnx] [--data training/dataset] [--assets training/assets]`; exits 1 if per-square board accuracy < 0.995 (the spec gate).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_training.py`:

```python
def test_eval_boards_perfect_on_fake_recognizer(tmp_path, monkeypatch):
    """eval_boards ground-truth plumbing: with the recognizer mocked to return the
    true grid, accuracy must be exactly 1.0."""
    import chess

    from training import evaluate as ev

    class PerfectRecognizer:
        def __init__(self, *a, **k):
            pass

        def classify_squares(self, img):
            grid = ev.CURRENT_TRUE_GRID
            import numpy as np

            return grid, np.ones((8, 8), np.float32)

    monkeypatch.setattr(ev, "Recognizer", PerfectRecognizer)
    fake_assets = tmp_path / "assets" / "lichess" / "kosal"
    fake_assets.mkdir(parents=True)
    from training.fetch_assets import PIECE_CODES

    for code in PIECE_CODES:
        img = Image.new("RGBA", (64, 64), (200, 0, 0, 255))
        img.save(fake_assets.parent / "kosal" / f"{code.lower()}.png")
    result = ev.eval_boards("unused.onnx", tmp_path / "assets", n_positions=3, seed=1)
    assert result["per_square"] == 1.0
    assert result["exact_boards"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_training.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.evaluate'`

- [ ] **Step 3: Write the implementation**

`training/evaluate.py`:

```python
import argparse
import random
import sys
from pathlib import Path

import chess
import cv2
import numpy as np

from chessbot.vision.position import CLASSES, board_to_grid
from chessbot.vision.recognizer import Recognizer
from training.generate_dataset import FLAT_THEMES, VAL_SETS, load_piece_set, render_full_board

CURRENT_TRUE_GRID = None


def eval_squares(model_path, data_root: Path) -> dict:
    recognizer = Recognizer(model_path=model_path)
    net = recognizer._net
    per_class = {}
    correct_total = count_total = 0
    for idx, cls in enumerate(CLASSES):
        paths = sorted((Path(data_root) / "val" / cls).glob("*.png"))
        if not paths:
            continue
        imgs = np.stack([cv2.imread(str(p)) for p in paths])
        blob = cv2.dnn.blobFromImages(imgs, 1 / 255.0, (48, 48), swapRB=False)
        net.setInput(blob)
        pred = net.forward().argmax(axis=1)
        correct = int((pred == idx).sum())
        per_class[cls] = correct / len(paths)
        correct_total += correct
        count_total += len(paths)
    return {"overall": correct_total / max(1, count_total), "per_class": per_class}


def _random_position(rng: random.Random) -> chess.Board:
    board = chess.Board()
    for _ in range(rng.randrange(6, 60)):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            break
        board.push(rng.choice(moves))
    return board


def eval_boards(model_path, assets_dir: Path, n_positions: int = 40, seed: int = 0) -> dict:
    global CURRENT_TRUE_GRID
    rng = random.Random(seed)
    recognizer = Recognizer(model_path=model_path)
    holdout_dirs = [
        d
        for source in ("lichess", "chesscom")
        for d in sorted((Path(assets_dir) / source).glob("*"))
        if d.is_dir() and d.name in VAL_SETS
    ]
    if not holdout_dirs:
        raise SystemExit(f"no holdout piece sets found under {assets_dir}")
    piece_sets = [load_piece_set(d) for d in holdout_dirs]
    square_hits = squares_total = exact = 0
    for _ in range(n_positions):
        board = _random_position(rng)
        white_at_bottom = rng.random() < 0.5
        true_grid = board_to_grid(board, white_at_bottom)
        CURRENT_TRUE_GRID = true_grid
        img = render_full_board(rng.choice(piece_sets), board, rng.choice(FLAT_THEMES),
                                square_px=rng.choice([48, 64, 80]), white_at_bottom=white_at_bottom)
        grid, _ = recognizer.classify_squares(img)
        hits = sum(grid[r][c] == true_grid[r][c] for r in range(8) for c in range(8))
        square_hits += hits
        squares_total += 64
        exact += hits == 64
    return {"per_square": square_hits / squares_total, "exact_boards": exact / n_positions}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the piece classifier")
    parser.add_argument("--model", type=Path, default=Path("models/piece_classifier.onnx"))
    parser.add_argument("--data", type=Path, default=Path("training/dataset"))
    parser.add_argument("--assets", type=Path, default=Path("training/assets"))
    parser.add_argument("--positions", type=int, default=40)
    args = parser.parse_args()

    squares = eval_squares(args.model, args.data)
    print(f"val squares overall: {squares['overall']:.4%}")
    for cls, acc in sorted(squares["per_class"].items(), key=lambda kv: kv[1]):
        print(f"  {cls}: {acc:.4%}")
    boards = eval_boards(args.model, args.assets, args.positions)
    print(f"held-out boards: per-square {boards['per_square']:.4%}, exact {boards['exact_boards']:.2%}")
    if boards["per_square"] < 0.995:
        sys.exit("FAIL: per-square accuracy below 99.5% gate")
    print("PASS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_training.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add training/evaluate.py tests/test_training.py
git commit -m "feat: classifier evaluation with held-out board gate"
```

---

### Task 18: Train the real model, commit weights + fixtures + integration test

This task runs the pipeline for real (network + ~10-30 min training on this machine) and makes the model a repo artifact.

**Files:**
- Create: `models/piece_classifier.onnx`, `tests/fixtures/board_startpos_cburnett.png`, `tests/fixtures/board_midgame_merida.png`, `tests/fixtures/board_endgame_alpha.png`, `tests/test_recognizer_model.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: entire training pipeline (T14-T17), `Recognizer` (T9), `grid_to_board`/`board_to_grid` (T3).
- Produces: the committed model consumed by `Settings.model_path` (T7) and `cmd_play` (T13).

- [ ] **Step 1: Fetch assets and generate the dataset**

Run:
```bash
python -m training.fetch_assets
python -m training.generate_dataset --per-class 160
```
Expected: counts printed, no more than a handful of failed URLs (missing sets are skipped with a warning); `training/dataset/train/` has 13 class dirs; `git status` shows NO new untracked files under `training/` (gitignore working).

- [ ] **Step 2: Train and evaluate**

Run:
```bash
python -m training.train --epochs 12
python -m training.evaluate
```
Expected: training prints rising val accuracy; evaluate prints per-square ≥ 99.5% and `PASS`. If the gate fails, inspect the worst classes from the per-class report, increase `--per-class` to 240 and epochs to 20, and retrain — do not lower the gate.

- [ ] **Step 3: Un-ignore the model and commit it**

Edit `.gitignore`: delete the `models/` line and the `*.onnx` line. Then:

```bash
git add .gitignore models/piece_classifier.onnx
git commit -m "feat: trained piece classifier model"
```

- [ ] **Step 4: Generate committed fixtures and write the integration test**

Run this snippet to create fixtures (lichess-derived renders only — no chess.com assets in the repo):

```bash
python - <<'EOF'
from pathlib import Path
import chess
from training.generate_dataset import FLAT_THEMES, load_piece_set, render_full_board
import cv2

fixtures = Path("tests/fixtures")
fixtures.mkdir(exist_ok=True)
cases = [
    ("board_startpos_cburnett.png", "cburnett", chess.STARTING_FEN, FLAT_THEMES[0], True),
    ("board_midgame_merida.png", "merida",
     "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 5 9", FLAT_THEMES[1], True),
    ("board_endgame_alpha.png", "alpha", "8/5pk1/6p1/8/3K4/8/5PP1/8 b - - 0 40", FLAT_THEMES[2], False),
]
for name, piece_set, fen, theme, white_bottom in cases:
    pieces = load_piece_set(Path("training/assets/lichess") / piece_set)
    img = render_full_board(pieces, chess.Board(fen), theme, square_px=64, white_at_bottom=white_bottom)
    cv2.imwrite(str(fixtures / name), img)
print("fixtures written")
EOF
```

`tests/test_recognizer_model.py`:

```python
from pathlib import Path

import chess
import cv2
import pytest

from chessbot.config import Settings
from chessbot.vision.position import board_to_grid
from chessbot.vision.recognizer import Recognizer

MODEL = Settings().model_path
FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="trained model not present")

CASES = [
    ("board_startpos_cburnett.png", chess.STARTING_FEN, True),
    ("board_midgame_merida.png", "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 5 9", True),
    ("board_endgame_alpha.png", "8/5pk1/6p1/8/3K4/8/5PP1/8 b - - 0 40", False),
]


@pytest.mark.parametrize("filename,fen,white_at_bottom", CASES)
def test_recognizer_reads_fixture_exactly(filename, fen, white_at_bottom):
    img = cv2.imread(str(FIXTURES / filename))
    assert img is not None
    grid, confidence = Recognizer(model_path=MODEL).classify_squares(img)
    assert grid == board_to_grid(chess.Board(fen), white_at_bottom)
    assert float(confidence.min()) > 0.5
```

- [ ] **Step 5: Run the integration test and full suite, then commit**

Run: `pytest tests/test_recognizer_model.py -v && pytest -q`
Expected: 3 passed on the model test; full suite green.

```bash
git add tests/fixtures tests/test_recognizer_model.py
git commit -m "test: recognizer integration fixtures against trained model"
```

---

### Task 19: README rewrite, legacy deletion, final sweep

**Files:**
- Modify: `README.md` (rename from `readme.md`, full rewrite below)
- Delete: `bot-offline.py`, `bot-online.py`, `bot_server.py`, `main.py`, `setup.py`, `test.py`

**Interfaces:**
- Consumes: everything — this is the final gate.

- [ ] **Step 1: Delete legacy scripts**

```bash
git rm bot-offline.py bot-online.py bot_server.py main.py setup.py test.py
git mv readme.md README.md
```

- [ ] **Step 2: Rewrite README.md**

Replace the entire content of `README.md` with:

````markdown
# OpenCV Chess Bot

A screen-reading chess bot. It finds the chessboard on your screen, recognizes
the pieces with a small CNN, and plays moves through a UCI engine (Stockfish,
Leela, ...) by clicking the squares — on lichess, chess.com, or anything that
looks like a 2D board.

https://user-images.githubusercontent.com/78639550/225330750-d877a4cf-8dda-4dcf-9b6c-3c035333fe6a.mp4

## How it works

1. **Board detection** — a gradient/Hough sweep over a screenshot finds the
   8×8 grid (`chessbot/vision/board_detect.py`).
2. **Piece recognition** — each square is classified by a ~55K-parameter CNN
   (13 classes: 6 pieces × 2 colors + empty) exported to ONNX and run with
   OpenCV's DNN module. No ML framework needed at runtime
   (`chessbot/vision/recognizer.py`). This is how the bot can join a game
   from **any position**, not just move 1.
3. **Move watching** — square-by-square frame diffing detects the opponent's
   move near-instantly (`chessbot/vision/move_detect.py`). If diffing ever
   gets confused, the recognizer re-reads the whole board and play continues.
4. **Engine** — moves come from an opening book for the first moves, then a
   UCI engine with configurable depth/time/humanized timing
   (`chessbot/engine/`).
5. **Clicking** — pyautogui clicks mapped through the capture scale (Retina
   safe). Promotions auto-pick the queen (`chessbot/control/mouse.py`).

Screenshots are captured natively: DirectX Desktop Duplication via **dxcam**
on Windows (sub-millisecond), **mss** (CoreGraphics) on macOS and Linux.

## Install

```bash
git clone https://github.com/7dpk/OpenCV-Chess-Bot && cd OpenCV-Chess-Bot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
pip install -e ".[windows]"    # Windows: adds dxcam
```

Install a UCI engine, e.g. Stockfish: `brew install stockfish` (macOS),
`winget install Stockfish.Stockfish` (Windows), or pass `--engine /path/to/engine`.

macOS: grant your terminal **Screen Recording** and **Accessibility**
permissions (System Settings → Privacy & Security).

## Play

Make the board fully visible on screen, then:

```bash
chessbot play                      # detects board, position, and your color
chessbot play --turn black        # side to move, when joining mid-game
chessbot play --engine ./stockfish --threads 4 --hash 512
```

The bot plays from the bottom side of the board as displayed.

### Timing options

```bash
chessbot play --tc "40/5m"                                  # classical: 40 moves in 5 minutes
chessbot play --depth-mode --depth 14                       # fixed depth
chessbot play --timing-mode delay  --timing-min 1 --timing-max 3   # human-like pacing
chessbot play --timing-mode engine --timing-min 5 --timing-max 10  # engine thinks 5-10s
chessbot play --timing-mode both   --timing-min 5 --timing-max 10  # both of the above
```

Defaults live in `default.ini`.

### Remote engine

Offload engine computation to another machine:

```bash
chessbot server --port 6751                 # on the strong machine
chessbot play --remote 1.2.3.4:6751         # on the playing machine
```

## Training the piece classifier

The committed model (`models/piece_classifier.onnx`) covers lichess and
chess.com themes. To retrain or extend:

```bash
pip install -e ".[training]"
python -m training.fetch_assets        # downloads piece sets (not committed)
python -m training.generate_dataset    # synthesizes labeled square crops
python -m training.train               # trains + exports ONNX
python -m training.evaluate            # held-out-theme accuracy gate (99.5%)
```

Training data is 100% synthetic: piece sets rendered over board themes with
scale/offset jitter, move highlights, blur, and JPEG artifacts. Validation
uses piece sets held out from training, so the reported accuracy reflects
unseen themes.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Package layout: `capture/` (screenshots), `vision/` (detection + recognition),
`engine/` (UCI, book, server), `control/` (mouse), `game.py` (main loop),
`cli.py` (entry point). Training code lives in `training/` and is never
imported at runtime.

## Disclaimer

Using engine assistance in rated online games violates lichess and chess.com
terms of service. Use this bot against other bots, for analysis, or in
contexts where automation is allowed.
````

- [ ] **Step 3: Full verification**

Run:
```bash
pytest -q
chessbot --help
chessbot play --help
python -c "import chessbot.game, chessbot.cli, chessbot.engine.server"
git status
```
Expected: full suite green, help output lists all flags, imports clean, `git status` shows only the intended deletions/renames staged.

- [ ] **Step 4: Verify no stray references to deleted files**

Run: `grep -rn "bot-offline\|bot_server\|bot-online\|d3dshot\|keyboard" --include="*.py" --include="*.toml" --include="*.md" . | grep -v docs/superpowers | grep -v .venv`
Expected: no hits (plan/spec docs excluded).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs: rewrite README for package layout; remove legacy scripts"
```

---

## Self-Review Notes

- **Spec coverage:** package restructure (T1, T19), capture layer with Retina handling (T6), board detection port with error-handling fixes (T4), move detection port without exits (T5), recognizer + CLASSES contract (T3, T9), FEN completion heuristics (T3, T11), resync fallback (T11), timing modes/book/depth heuristic preserved (T8, T11), server + remote (T12), CLI flags per spec (T13), training pipeline (T14-T17), model artifact + integration tests (T18), README + cleanup (T19). Low-confidence confirm gate: T11 `read_position`. Engine-missing message: T8 `find_engine`.
- **Deviations from spec (intentional):** `CLASSES` lives in `vision/position.py` rather than `recognizer.py` (recognizer imports it; avoids a cv2 import for training's label list). `engine/uci.py` combines wrapper+book+timing (was listed as one file in spec anyway). Promotion auto-queens instead of prompting (spec's terminal-prompt fallback retained in `_execute_move`).
- **Type consistency check:** `best_move(board, *, depth=None, move_time=None) -> str` is the shared engine duck-type across `EngineClient`, `RemoteEngine`, `FakeEngine`, and `compute_move`. `Region = (left, top, right, bottom)` capture pixels everywhere. `grid[row][col]` row-0-top everywhere. `white_at_bottom: bool` everywhere (never "are_we_white").
- **Known risks:** (1) chess.com CDN URL scheme may change — fetch tolerates failures and training proceeds with lichess sets alone; the accuracy gate still applies. (2) The synthetic board-detect test exercises ported math; if tolerance fails, compare intermediate values against the original `bot-offline.py` implementation rather than tuning thresholds. (3) mss point/pixel behavior differs across OS versions — `scale` is measured empirically from a real grab, never assumed.




