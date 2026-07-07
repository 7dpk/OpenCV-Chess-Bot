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
