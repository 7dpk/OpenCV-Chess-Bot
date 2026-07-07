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
