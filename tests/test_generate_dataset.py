import random

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw

from chessbot.vision.position import CLASSES
from training.fetch_assets import PIECE_CODES
from training.generate_dataset import (
    FLAT_THEMES,
    MIN_SQUARE_PX,
    _draw_cursor,
    _draw_move_marker,
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
    import cv2

    pieces = load_piece_set(fake_set_dir / "chesscom" / "fake")
    rng = random.Random(0)
    light, dark = FLAT_THEMES[0]
    occupied = render_square(pieces["wK"], light, dark, True, rng)
    empty = render_square(None, light, dark, False, rng)
    assert occupied.shape == (48, 48, 3) and occupied.dtype == np.uint8
    assert empty.shape == (48, 48, 3)
    # augmented empty squares may legitimately contain edge energy (boundary
    # bleed, markers, cursor), so compare the piece silhouette on clean renders:
    # spatial detail (edge energy) rather than raw std, which background color
    # spread can dominate.
    clean_occupied = render_square(pieces["wK"], light, dark, True, rng, augment=False)
    clean_empty = render_square(None, light, dark, False, rng, augment=False)
    detail = lambda img: cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert detail(clean_occupied) > detail(clean_empty)


def test_render_square_deterministic_with_seed(fake_set_dir):
    pieces = load_piece_set(fake_set_dir / "chesscom" / "fake")
    light, dark = FLAT_THEMES[0]
    a = render_square(pieces["bN"], light, dark, True, random.Random(42))
    b = render_square(pieces["bN"], light, dark, True, random.Random(42))
    assert np.array_equal(a, b)


def test_min_square_px_covers_small_boards():
    # real boards can be as small as ~32px per square on screen; training must
    # render below that so the upscale artifacts are part of the distribution
    assert MIN_SQUARE_PX <= 32


def test_render_square_with_neighbors_deterministic(fake_set_dir):
    pieces = load_piece_set(fake_set_dir / "chesscom" / "fake")
    neighbors = list(pieces.values())
    light, dark = FLAT_THEMES[0]
    a = render_square(pieces["bN"], light, dark, True, random.Random(7), neighbors=neighbors)
    b = render_square(pieces["bN"], light, dark, True, random.Random(7), neighbors=neighbors)
    assert a.shape == (48, 48, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)


def test_draw_move_marker_changes_pixels():
    for occupied in (False, True):
        canvas = Image.new("RGBA", (64, 64), (100, 120, 80, 255))
        before = np.asarray(canvas.convert("RGB")).copy()
        _draw_move_marker(canvas, random.Random(0), occupied=occupied)
        after = np.asarray(canvas.convert("RGB"))
        assert not np.array_equal(before, after)
        assert after.shape == before.shape


def test_draw_cursor_changes_pixels():
    canvas = Image.new("RGBA", (64, 64), (100, 120, 80, 255))
    before = np.asarray(canvas.convert("RGB")).copy()
    _draw_cursor(canvas, random.Random(0))
    assert not np.array_equal(before, np.asarray(canvas.convert("RGB")))


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
