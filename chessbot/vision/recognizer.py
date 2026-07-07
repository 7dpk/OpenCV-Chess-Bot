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
