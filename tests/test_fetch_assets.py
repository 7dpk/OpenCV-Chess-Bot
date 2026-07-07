import sys

import training.fetch_assets as fetch_assets_module
from training.fetch_assets import (
    CHESSCOM_PIECE_SETS,
    LICHESS_PIECE_SETS,
    PIECE_CODES,
    chesscom_piece_url,
    lichess_board_url,
    lichess_piece_url,
)


def test_piece_codes():
    assert len(PIECE_CODES) == 12
    assert PIECE_CODES[0] == "wP" and PIECE_CODES[-1] == "bK"


def test_set_lists_nonempty():
    assert len(LICHESS_PIECE_SETS) >= 25
    assert "cburnett" in LICHESS_PIECE_SETS
    assert len(CHESSCOM_PIECE_SETS) >= 5
    assert "neo" in CHESSCOM_PIECE_SETS


def test_lichess_piece_url():
    assert (
        lichess_piece_url("cburnett", "wP")
        == "https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett/wP.svg"
    )


def test_chesscom_piece_url_lowercases():
    assert (
        chesscom_piece_url("neo", "bQ")
        == "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bq.png"
    )


def test_lichess_board_url():
    assert (
        lichess_board_url("wood.jpg")
        == "https://raw.githubusercontent.com/lichess-org/lila/master/public/images/board/wood.jpg"
    )


def test_main_survives_total_download_failure(tmp_path, monkeypatch):
    """main() must not crash with FileNotFoundError when every download fails
    and dest/lichess is never created."""
    empty_dest = tmp_path / "assets"

    def fake_fetch_all(dest):
        assert dest == empty_dest
        return {"lichess": 0, "chesscom": 0, "boards": 0, "failed": 999}

    monkeypatch.setattr(fetch_assets_module, "fetch_all", fake_fetch_all)
    monkeypatch.setattr(sys, "argv", ["fetch_assets.py", "--dest", str(empty_dest)])

    assert not (empty_dest / "lichess").exists()
    fetch_assets_module.main()  # should not raise
