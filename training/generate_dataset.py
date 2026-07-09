import argparse
import io
import random
from pathlib import Path

import chess
import cv2
import numpy as np
from PIL import Image, ImageDraw

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

# real boards can be as small as ~32px per square on screen; rendering below that
# puts the upscale-to-48 artifacts inside the training distribution
MIN_SQUARE_PX = 28


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


def _draw_move_marker(canvas: Image.Image, rng: random.Random, occupied: bool) -> None:
    """Translucent legal-move marker as chess sites draw on move targets:
    a centred dot on empty squares, a capture ring on occupied ones."""
    size = canvas.size[0]
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    shade = rng.randrange(0, 90)
    color = (shade, shade, shade, rng.randrange(50, 110))
    if occupied:
        width = max(2, int(size * rng.uniform(0.06, 0.10)))
        draw.ellipse([1, 1, size - 2, size - 2], outline=color, width=width)
    else:
        radius = size * rng.uniform(0.14, 0.20)
        c0, c1 = size / 2 - radius, size / 2 + radius
        draw.ellipse([c0, c0, c1, c1], fill=color)
    canvas.alpha_composite(overlay)


def _draw_cursor(canvas: Image.Image, rng: random.Random) -> None:
    """Mouse pointer resting on a square (it stays wherever the last move dropped it)."""
    size = canvas.size[0]
    s = size * rng.uniform(0.20, 0.35) / 19.0
    x = rng.uniform(0.1, 0.7) * size
    y = rng.uniform(0.1, 0.6) * size
    white_body = rng.random() < 0.5
    body = (255, 255, 255, 255) if white_body else (20, 20, 20, 255)
    outline = (0, 0, 0, 255) if white_body else (240, 240, 240, 255)
    points = [(x, y), (x, y + 16 * s), (x + 4 * s, y + 12 * s), (x + 7 * s, y + 19 * s),
              (x + 9 * s, y + 18 * s), (x + 6 * s, y + 11 * s), (x + 11 * s, y + 11 * s)]
    ImageDraw.Draw(canvas).polygon(points, fill=body, outline=outline)


def _center_cell(canvas: Image.Image, pad: int, size: int, fn) -> None:
    cell = canvas.crop((pad, pad, pad + size, pad + size))
    fn(cell)
    canvas.paste(cell, (pad, pad))


def render_square(piece_img, light_bg, dark_bg, is_light: bool, rng: random.Random,
                  out_size: int = 48, augment: bool = True, neighbors=None) -> np.ndarray:
    size = rng.randrange(MIN_SQUARE_PX, 129) if augment else 96
    # render the square inside a border of opposite-colour background, then crop
    # with a random offset: this bakes board-detect misalignment (square-boundary
    # bleed, neighbour piece slivers) into the training distribution
    pad = max(3, size // 8) if augment else 0
    canvas = _background(light_bg, dark_bg, not is_light, size + 2 * pad)
    canvas.paste(_background(light_bg, dark_bg, is_light, size), (pad, pad))
    if augment and rng.random() < 0.30:
        overlay = Image.new("RGBA", (size, size), rng.choice(HIGHLIGHTS))
        _center_cell(canvas, pad, size, lambda cell: cell.alpha_composite(overlay))
    if augment and neighbors and rng.random() < 0.35:
        neighbor_px = max(8, int(size * rng.uniform(0.78, 0.97)))
        inset = (size - neighbor_px) // 2
        neighbor = rng.choice(neighbors).resize((neighbor_px, neighbor_px), Image.LANCZOS)
        side = rng.randrange(4)
        nx, ny = [(pad + inset, pad - size + inset), (pad + inset, pad + size + inset),
                  (pad - size + inset, pad + inset), (pad + size + inset, pad + inset)][side]
        canvas.paste(neighbor, (nx, ny), neighbor)
    if piece_img is not None:
        scale = rng.uniform(0.75, 0.97) if augment else 0.85
        piece_px = max(8, int(size * scale))
        piece = piece_img.resize((piece_px, piece_px), Image.LANCZOS)
        max_off = int(size * 0.06) if augment else 0
        ox = pad + (size - piece_px) // 2 + (rng.randint(-max_off, max_off) if augment else 0)
        oy = pad + (size - piece_px) // 2 + (rng.randint(-max_off, max_off) if augment else 0)
        canvas.alpha_composite(piece, (ox, oy))
    if augment:
        if rng.random() < (0.10 if piece_img is not None else 0.15):
            _center_cell(canvas, pad, size,
                         lambda cell: _draw_move_marker(cell, rng, piece_img is not None))
        if rng.random() < 0.08:
            _center_cell(canvas, pad, size, lambda cell: _draw_cursor(cell, rng))
        # triangular: most crops nearly aligned (like real board detection), with
        # occasional large misalignment in the tails
        dx, dy = round(rng.triangular(-pad, pad, 0)), round(rng.triangular(-pad, pad, 0))
        canvas = canvas.crop((pad + dx, pad + dy, pad + dx + size, pad + dy + size))
    else:
        canvas = canvas.crop((pad, pad, pad + size, pad + size))
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
        neighbors = list(pieces.values())
        for label in ["empty", *PIECE_CODES]:
            class_dir = out_dir / split / label
            class_dir.mkdir(parents=True, exist_ok=True)
            piece_img = None if label == "empty" else pieces[label]
            for i in range(n):
                light, dark = rng.choice(backgrounds)
                crop = render_square(piece_img, light, dark, rng.random() < 0.5, rng,
                                     neighbors=neighbors)
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
