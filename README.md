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
```

On macOS, `cairosvg` needs Homebrew's cairo: `brew install cairo`, and if
rendering fails with a dylib error, run with
`DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`.

```bash
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
