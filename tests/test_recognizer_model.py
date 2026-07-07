from pathlib import Path

import chess
import cv2
import pytest

from chessbot.config import Settings
from chessbot.vision.position import board_to_grid
from chessbot.vision.recognizer import Recognizer

MODEL = Settings().model_path
FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="trained model not present")

CASES = [
    ("board_startpos_cburnett.png", chess.STARTING_FEN, True),
    ("board_midgame_merida.png", "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 5 9", True),
    ("board_endgame_alpha.png", "8/5pk1/6p1/8/3K4/8/5PP1/8 b - - 0 40", False),
]


@pytest.mark.parametrize("filename,fen,white_at_bottom", CASES)
def test_recognizer_reads_fixture_exactly(filename, fen, white_at_bottom):
    img = cv2.imread(str(FIXTURES / filename))
    assert img is not None
    grid, confidence = Recognizer(model_path=MODEL).classify_squares(img)
    assert grid == board_to_grid(chess.Board(fen), white_at_bottom)
    assert float(confidence.min()) > 0.5
