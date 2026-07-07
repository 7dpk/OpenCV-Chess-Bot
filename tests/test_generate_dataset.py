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
    import cv2

    pieces = load_piece_set(fake_set_dir / "chesscom" / "fake")
    rng = random.Random(0)
    light, dark = FLAT_THEMES[0]
    occupied = render_square(pieces["wK"], light, dark, True, rng)
    empty = render_square(None, light, dark, False, rng)
    assert occupied.shape == (48, 48, 3) and occupied.dtype == np.uint8
    assert empty.shape == (48, 48, 3)
    # a flat empty square can have a larger raw pixel std than a piece square
    # purely from cross-channel background color spread, so compare spatial
    # detail (edge energy) rather than raw std to detect the piece silhouette.
    detail = lambda img: cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert detail(occupied) > detail(empty)


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
