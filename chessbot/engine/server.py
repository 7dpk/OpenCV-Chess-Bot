import socket
import threading

import chess

ENCODING = "utf-8"
TABLEBASE_URL = "http://tablebase.lichess.ovh/standard"


def _tablebase_move(fen: str) -> str | None:
    if len(chess.Board(fen).piece_map()) >= 8:
        return None
    try:
        import requests

        response = requests.get(TABLEBASE_URL, params={"fen": fen.replace(" ", "_")}, timeout=5)
        if response.status_code == 200:
            return response.json()["moves"][0]["uci"]
    except Exception:
        return None
    return None


def compute_move(request: str, engine) -> str:
    try:
        fen, tc, limit = (part.strip() for part in request.split(","))
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"malformed request: {request!r}") from exc
    move = _tablebase_move(fen)
    if move is not None:
        return move
    if tc == "depth":
        return engine.best_move(board, depth=int(limit))
    return engine.best_move(board, move_time=float(limit))


def _handle_client(conn: socket.socket, engine) -> None:
    with conn:
        while True:
            try:
                data = conn.recv(128)
            except ConnectionResetError:
                return
            if not data:
                return
            try:
                move = compute_move(data.decode(ENCODING), engine)
            except ValueError as exc:
                print(f"Dropping client: {exc}")
                return
            conn.sendall(move.encode(ENCODING))
            print(f"Sent move: {move}")


def serve(engine, host: str = "0.0.0.0", port: int = 6751) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen()
        print(f"Serving moves on {host}:{port}")
        while True:
            conn, addr = server.accept()
            print(f"Connected: {addr}")
            threading.Thread(target=_handle_client, args=(conn, engine), daemon=True).start()
