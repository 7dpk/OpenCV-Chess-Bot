import argparse
import random
import sys
from pathlib import Path

import chess
import cv2
import numpy as np
from PIL import Image

from chessbot.vision.position import CLASSES, board_to_grid
from chessbot.vision.recognizer import Recognizer
from training.generate_dataset import (
    FLAT_THEMES,
    VAL_SETS,
    _draw_cursor,
    _draw_move_marker,
    load_piece_set,
    render_full_board,
)

CURRENT_TRUE_GRID = None

CONFIDENCE_FLOOR = 0.90
SOFT_FLOOR = 0.80  # mirror of GameSession.RESYNC_SOFT_FLOOR: best-effort adoption tier

# what most players actually see: the site default piece sets (all trained on)
DEPLOYED_SETS = ["cburnett", "merida", "alpha", "neo", "classic"]


def _shift_crop(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Re-crop with a small offset, mimicking board-detect alignment error."""
    if dx == 0 and dy == 0:
        return img
    m = max(abs(dx), abs(dy))
    big = cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_REFLECT)
    h, w = img.shape[:2]
    return big[m + dy : m + dy + h, m + dx : m + dx + w]


def apply_screen_artifacts(img: np.ndarray, true_grid, sq_px: int, rng: random.Random) -> np.ndarray:
    """Overlay UI artifacts a live chess site adds on top of the raw board:
    legal-move markers on a few empty squares and a mouse cursor."""
    canvas = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")

    def on_cell(row, col, fn):
        box = (col * sq_px, row * sq_px, (col + 1) * sq_px, (row + 1) * sq_px)
        cell = canvas.crop(box)
        fn(cell)
        canvas.paste(cell, box[:2])

    empties = [(r, c) for r in range(8) for c in range(8) if true_grid[r][c] == "empty"]
    rng.shuffle(empties)
    for row, col in empties[: rng.randrange(0, 4)]:
        on_cell(row, col, lambda cell: _draw_move_marker(cell, rng, occupied=False))
    on_cell(rng.randrange(8), rng.randrange(8), lambda cell: _draw_cursor(cell, rng))
    return cv2.cvtColor(np.asarray(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)


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


def eval_boards(model_path, assets_dir: Path, n_positions: int = 40, seed: int = 0,
                set_names=None) -> dict:
    global CURRENT_TRUE_GRID
    rng = random.Random(seed)
    recognizer = Recognizer(model_path=model_path)
    wanted = VAL_SETS if set_names is None else set(set_names)
    set_dirs = [
        d
        for source in ("lichess", "chesscom")
        for d in sorted((Path(assets_dir) / source).glob("*"))
        if d.is_dir() and d.name in wanted
    ]
    if not set_dirs:
        raise SystemExit(f"no matching piece sets found under {assets_dir}")
    piece_sets = [load_piece_set(d) for d in set_dirs]
    square_hits = squares_total = exact = floor_clear = 0
    usable_main = n_main = misleading = 0
    for _ in range(n_positions):
        board = _random_position(rng)
        white_at_bottom = rng.random() < 0.5
        true_grid = board_to_grid(board, white_at_bottom)
        CURRENT_TRUE_GRID = true_grid
        sq_px = rng.choice([32, 48, 64, 80])
        img = render_full_board(rng.choice(piece_sets), board, rng.choice(FLAT_THEMES),
                                square_px=sq_px, white_at_bottom=white_at_bottom)
        img = apply_screen_artifacts(img, true_grid, sq_px, rng)
        img = _shift_crop(img, rng.randint(-3, 3), rng.randint(-3, 3))
        grid, confidence = recognizer.classify_squares(img)
        hits = sum(grid[r][c] == true_grid[r][c] for r in range(8) for c in range(8))
        square_hits += hits
        squares_total += 64
        exact += hits == 64
        min_conf = float(np.min(confidence))
        floor_clear += min_conf >= CONFIDENCE_FLOOR
        misleading += min_conf >= SOFT_FLOOR and hits != 64
        if sq_px >= 48:  # the supported size range: real boards render >=48px squares
            n_main += 1
            usable_main += min_conf >= SOFT_FLOOR and hits == 64
    return {
        "per_square": square_hits / squares_total,
        "exact_boards": exact / n_positions,
        "floor_clear": floor_clear / n_positions,
        "usable_main": usable_main / max(1, n_main),
        "misleading": misleading / n_positions,
    }


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
    print(
        f"held-out boards: per-square {boards['per_square']:.4%}, exact {boards['exact_boards']:.2%}, "
        f"clearing {CONFIDENCE_FLOOR} confidence floor {boards['floor_clear']:.2%}"
    )
    print(
        f"  usable at soft floor {SOFT_FLOOR} (>=48px squares): {boards['usable_main']:.2%}, "
        f"adoptable-but-wrong: {boards['misleading']:.2%}"
    )
    deployed = eval_boards(args.model, args.assets, args.positions, set_names=DEPLOYED_SETS)
    print(
        f"deployed-style boards (site defaults): per-square {deployed['per_square']:.4%}, "
        f"clearing floor {deployed['floor_clear']:.2%}, adoptable-but-wrong {deployed['misleading']:.2%}"
    )
    if boards["per_square"] < 0.995:
        sys.exit("FAIL: held-out per-square accuracy below 99.5% gate")
    if boards["misleading"] > 0.01:
        sys.exit(f"FAIL: more than 1% of held-out boards adoptable at {SOFT_FLOOR} but misread")
    if deployed["floor_clear"] < 0.95:
        sys.exit(f"FAIL: fewer than 95% of deployed-style boards clear the {CONFIDENCE_FLOOR} floor")
    if deployed["misleading"] > 0.01:
        sys.exit("FAIL: more than 1% of deployed-style boards adoptable but misread")
    print("PASS")


if __name__ == "__main__":
    main()
