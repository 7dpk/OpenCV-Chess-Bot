import socket

import chess

from .server import ENCODING


class RemoteEngine:
    def __init__(self, host: str, port: int):
        self._sock = socket.create_connection((host, port), timeout=60)

    def best_move(self, board: chess.Board, *, depth: int | None = None, move_time: float | None = None) -> str:
        if depth is not None:
            request = f"{board.fen()},depth,{depth}"
        else:
            request = f"{board.fen()},time,{move_time if move_time else 0.1}"
        self._sock.sendall(request.encode(ENCODING))
        return self._sock.recv(128).decode(ENCODING)

    def close(self) -> None:
        self._sock.close()
