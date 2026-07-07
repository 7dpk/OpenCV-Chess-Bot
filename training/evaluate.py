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
