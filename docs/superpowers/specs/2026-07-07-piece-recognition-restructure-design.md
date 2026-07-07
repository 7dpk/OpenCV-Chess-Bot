# Piece Recognition + Repo Restructure — Design

Date: 2026-07-07
Status: Approved

## Goal

Extend the bot to recognize pieces on the board so it can join a game from any
position (today it must start from move 1), make screenshot capture native and
fast on Windows and macOS, and restructure the repo into an installable,
readable package.

## Non-goals

- Arbitrary desktop chess GUIs (Arena, ChessBase). The recognizer targets
  lichess and chess.com themes; other UIs may work via generalization but are
  not guaranteed.
- Variants (Chess960 castling, antichess, etc.).
- GUI. The Tk GUI (`main.py`) is removed; CLI only.

## Architecture

```
chessbot/                     runtime package (no ML framework dependency)
  cli.py                      `chessbot play` / `chessbot server`; keeps all
                              existing flags: --tc, --depth-mode, --threads,
                              --hash, --depth, --timing-mode/min/max, adds
                              --turn, --engine, --capture
  config.py                   default.ini loading + CLI override merging
  capture/
    base.py                   Capturer protocol: grab(region) -> np.ndarray (BGR)
    mss_backend.py            all platforms; native CoreGraphics on macOS
    dxcam_backend.py          Windows only; Desktop Duplication API
    __init__.py               factory: Windows -> dxcam, fallback mss;
                              macOS/Linux -> mss. Handles Retina scaling map
                              between capture pixels and pyautogui points.
  vision/
    board_detect.py           existing gradient/Hough corner detection (cleaned)
    squares.py                square cropping, (row,col) <-> square-name math
    recognizer.py             ONNX CNN inference via cv2.dnn; board image ->
                              64 labels -> chess.Board placement
    move_detect.py            existing diff-based move detection (cleaned)
  engine/
    uci.py                    engine wrapper: options, book, timing modes
    server.py, remote.py      socket server + client (current online mode)
  control/
    mouse.py                  clicking, promotion handling (terminal prompt,
                              no `keyboard` dependency — it needs root on mac)
  game.py                     main play loop

training/                     dev-time only (torch, cairosvg, requests)
  fetch_assets.py             download lichess piece SVGs + board themes from
                              the lila repo, chess.com standard piece sets.
                              Assets are NOT committed (chess.com assets are
                              proprietary; only trained weights ship).
  generate_dataset.py         render labeled 48x48 square crops
  train.py                    PyTorch training, ONNX export
  evaluate.py                 accuracy on held-out themes + fixture screenshots

models/piece_classifier.onnx  committed trained model (~400 KB)
assets/Performance.bin        opening book
tests/                        pytest + fixture board images
```

Deleted: `bot.cp38-win_amd64.pyd`, `chess_bot-main.zip`, `main.py`, `test.py`,
vim `~`/`.un~` files, root-level `bot-offline.py`/`bot-online.py`/
`bot_server.py` (logic refactored into the package; history preserves them).

## Piece recognition

- 13 classes: {P,N,B,R,Q,K} x {white,black} + empty.
- Input 48x48 RGB square crop; small CNN (~100K params, 3 conv blocks + GAP + FC).
- Inference: all 64 squares as one batch through `cv2.dnn.readNetFromONNX` —
  few ms on CPU; torch is never a runtime dependency.
- Training data: 100% synthetic. Every lichess piece set (~60) and chess.com
  standard sets rendered onto light/dark square backgrounds from real board
  themes, augmented with: scale/offset jitter, last-move highlight overlays,
  check highlight, brightness/contrast, JPEG artifacts, slight blur.
- Validation on piece sets held out from training (proves theme generalization).
- Target: >99.5% per-square accuracy on held-out themes (a board read is 64
  squares; per-square errors compound).

## FEN completion (start from any position)

- Placement: from recognizer.
- Orientation/our color: whichever color's pieces dominate the bottom half of
  the board (current corner-brightness heuristic kept as fallback).
- Side to move: `--turn white|black`; if omitted and position is not the
  starting position, prompt once interactively.
- Castling rights: granted iff king and rook are on their home squares.
- En passant: assumed none.

## Game loop integration

The fast diff-based move detection continues to drive the game (it is
near-instant). The recognizer runs:

1. At startup, to build the initial FEN.
2. As a re-sync fallback: when diff detection yields no legal move (the
   current code exits the process here), re-recognize the full board, rebuild
   the position, and continue.

## Capture performance

- Windows: dxcam (Desktop Duplication) region grabs, sub-ms; mss fallback.
- macOS: mss (CoreGraphics), ~5-10 ms region grabs; Retina 2x scaling handled
  by mapping capture pixels -> screen points for clicks.
- Capturer is constructed once; region grabs reuse the session (no per-frame
  setup).

## Error handling

- Board not found: retry with guidance message instead of `os._exit`.
- Illegal/ambiguous diff move: re-sync via recognizer instead of exiting.
- Engine binary missing: clear message with `--engine` hint.
- Low recognizer confidence (softmax < threshold on any square): warn and ask
  user to confirm the printed FEN before play begins.

## Testing

- Unit: FEN building from known placements, castling heuristic, orientation
  inference, square math, config/CLI merging.
- Recognizer eval: held-out-theme accuracy report in `training/evaluate.py`.
- Integration: fixture full-board screenshots (synthetic renders + a few real
  lichess/chess.com captures) -> exact FEN match.
- Capture backends: smoke test grabbing a region and checking shape/dtype.

## Dependencies

- Runtime: opencv-python, numpy, python-chess, pyautogui, mss; dxcam
  (Windows extra). Removed: d3dshot, keyboard, Cython, tkinter.
- Training (optional extra): torch, torchvision, cairosvg, pillow, requests.
- Python >= 3.10.
