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


def test_train_resumes_from_checkpoint(tiny_dataset, tmp_path):
    ckpt = tmp_path / "train.ckpt"
    train_model(tiny_dataset, epochs=1, batch_size=8, device="cpu", ckpt_path=ckpt)
    assert ckpt.exists()
    # resuming with a larger epoch budget continues rather than restarting
    model, acc = train_model(tiny_dataset, epochs=2, batch_size=8, device="cpu", ckpt_path=ckpt)
    saved = torch.load(ckpt, weights_only=False)
    assert saved["epoch"] == 2
    assert 0.0 <= acc <= 1.0


def test_shift_crop_preserves_shape_and_shifts():
    from training.evaluate import _shift_crop

    img = np.zeros((256, 256, 3), np.uint8)
    img[0, 0] = 255
    out = _shift_crop(img, 3, -2)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)
    assert np.array_equal(_shift_crop(img, 0, 0), img)


def test_apply_screen_artifacts_keeps_shape_and_is_deterministic():
    from chessbot.vision.position import start_grid
    from training.evaluate import apply_screen_artifacts

    img = np.full((256, 256, 3), 120, np.uint8)
    grid = start_grid(True)
    a = apply_screen_artifacts(img.copy(), grid, 32, random.Random(3))
    b = apply_screen_artifacts(img.copy(), grid, 32, random.Random(3))
    assert a.shape == img.shape
    assert not np.array_equal(a, img)
    assert np.array_equal(a, b)


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
        img.save(fake_assets / f"{code.lower()}.png")
    result = ev.eval_boards("unused.onnx", tmp_path / "assets", n_positions=3, seed=1)
    assert result["per_square"] == 1.0
    assert result["exact_boards"] == 1.0
