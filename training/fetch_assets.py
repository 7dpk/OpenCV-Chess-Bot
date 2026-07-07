import argparse
from pathlib import Path

LILA_RAW = "https://raw.githubusercontent.com/lichess-org/lila/master/public"

LICHESS_PIECE_SETS = [
    "alpha", "anarcandy", "caliente", "california", "cardinal", "cburnett", "celtic",
    "chess7", "chessnut", "companion", "cooke", "dubrovny", "fantasy", "fresca",
    "gioco", "governor", "horsey", "icpieces", "kiwen-suwi", "kosal", "leipzig",
    "maestro", "merida", "monarchy", "mpchess", "pirouetti", "pixel", "reillycraig",
    "rhosgfx", "riohacha", "spatial", "staunty", "tatiana", "xkcd",
]

CHESSCOM_PIECE_SETS = ["neo", "classic", "wood", "glass", "gothic", "metal", "bases", "icy_sea"]

LICHESS_BOARD_IMAGES = [
    "wood.jpg", "wood2.jpg", "wood3.jpg", "wood4.jpg", "maple.jpg", "maple2.jpg",
    "marble.jpg", "grey.jpg", "metal.jpg", "olive.jpg", "blue-marble.jpg",
    "green-plastic.png", "leather.jpg", "canvas2.jpg", "horsey.jpg",
    "pink-pyramid.png", "purple-diag.png",
]

PIECE_CODES = ["wP", "wN", "wB", "wR", "wQ", "wK", "bP", "bN", "bB", "bR", "bQ", "bK"]


def lichess_piece_url(piece_set: str, piece: str) -> str:
    return f"{LILA_RAW}/piece/{piece_set}/{piece}.svg"


def chesscom_piece_url(piece_set: str, piece: str) -> str:
    return f"https://images.chesscomfiles.com/chess-themes/pieces/{piece_set}/150/{piece.lower()}.png"


def lichess_board_url(filename: str) -> str:
    return f"{LILA_RAW}/images/board/{filename}"


def _download(session, url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    response = session.get(url, timeout=30)
    if response.status_code != 200:
        print(f"skip ({response.status_code}): {url}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    return True


def fetch_all(dest: Path) -> dict:
    import requests

    session = requests.Session()
    counts = {"lichess": 0, "chesscom": 0, "boards": 0, "failed": 0}
    for piece_set in LICHESS_PIECE_SETS:
        for piece in PIECE_CODES:
            path = dest / "lichess" / piece_set / f"{piece}.svg"
            if _download(session, lichess_piece_url(piece_set, piece), path):
                counts["lichess"] += 1
            else:
                counts["failed"] += 1
    for piece_set in CHESSCOM_PIECE_SETS:
        for piece in PIECE_CODES:
            path = dest / "chesscom" / piece_set / f"{piece.lower()}.png"
            if _download(session, chesscom_piece_url(piece_set, piece), path):
                counts["chesscom"] += 1
            else:
                counts["failed"] += 1
    for filename in LICHESS_BOARD_IMAGES:
        if _download(session, lichess_board_url(filename), dest / "boards" / filename):
            counts["boards"] += 1
        else:
            counts["failed"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Download piece and board assets for training")
    parser.add_argument("--dest", type=Path, default=Path("training/assets"))
    args = parser.parse_args()
    counts = fetch_all(args.dest)
    print(counts)
    lichess_dir = args.dest / "lichess"
    incomplete = []
    if lichess_dir.is_dir():
        incomplete = [
            d.name
            for d in lichess_dir.iterdir()
            if d.is_dir() and len(list(d.glob("*.svg"))) != 12
        ]
    if incomplete:
        print(f"warning: incomplete lichess sets (will be skipped in training): {incomplete}")


if __name__ == "__main__":
    main()
