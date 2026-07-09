import chess

CLASSES = ["empty", "wP", "wN", "wB", "wR", "wQ", "wK", "bP", "bN", "bB", "bR", "bQ", "bK"]


def label_to_piece(label: str) -> chess.Piece | None:
    if label == "empty":
        return None
    symbol = label[1] if label[0] == "w" else label[1].lower()
    return chess.Piece.from_symbol(symbol)


def piece_to_label(piece: chess.Piece | None) -> str:
    if piece is None:
        return "empty"
    color = "w" if piece.color == chess.WHITE else "b"
    return color + piece.symbol().upper()


def _screen_to_square(row: int, col: int, white_at_bottom: bool) -> int:
    if white_at_bottom:
        return chess.square(col, 7 - row)
    return chess.square(7 - col, row)


def screen_to_square_name(row: int, col: int, white_at_bottom: bool) -> str:
    return chess.square_name(_screen_to_square(row, col, white_at_bottom))


def infer_white_at_bottom(grid: list[list[str]]) -> bool:
    score = 0
    for row in range(8):
        for col in range(8):
            label = grid[row][col]
            if label == "empty":
                continue
            bottom = row >= 4
            if label[0] == "w":
                score += 1 if bottom else -1
            else:
                score += -1 if bottom else 1
    return score >= 0


def board_to_grid(board: chess.Board, white_at_bottom: bool) -> list[list[str]]:
    return [
        [piece_to_label(board.piece_at(_screen_to_square(row, col, white_at_bottom))) for col in range(8)]
        for row in range(8)
    ]


def start_grid(white_at_bottom: bool) -> list[list[str]]:
    return board_to_grid(chess.Board(), white_at_bottom)


def _castling_rights(board: chess.Board) -> int:
    rights = 0
    if board.piece_at(chess.E1) == chess.Piece.from_symbol("K"):
        if board.piece_at(chess.H1) == chess.Piece.from_symbol("R"):
            rights |= chess.BB_H1
        if board.piece_at(chess.A1) == chess.Piece.from_symbol("R"):
            rights |= chess.BB_A1
    if board.piece_at(chess.E8) == chess.Piece.from_symbol("k"):
        if board.piece_at(chess.H8) == chess.Piece.from_symbol("r"):
            rights |= chess.BB_H8
        if board.piece_at(chess.A8) == chess.Piece.from_symbol("r"):
            rights |= chess.BB_A8
    return rights


def grid_to_board(grid: list[list[str]], white_at_bottom: bool, turn: chess.Color) -> chess.Board:
    board = chess.Board(None)
    for row in range(8):
        for col in range(8):
            piece = label_to_piece(grid[row][col])
            if piece is not None:
                board.set_piece_at(_screen_to_square(row, col, white_at_bottom), piece)
    board.turn = turn
    board.castling_rights = _castling_rights(board)
    _validate(board)
    return board


def _validate(board: chess.Board) -> None:
    for color in (chess.WHITE, chess.BLACK):
        if len(board.pieces(chess.KING, color)) != 1:
            raise ValueError("expected exactly one king per side; recognition is likely wrong")
    pawns = board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)
    if any(chess.square_rank(sq) in (0, 7) for sq in pawns):
        raise ValueError("pawn on a back rank; recognition is likely wrong")
