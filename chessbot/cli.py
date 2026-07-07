import argparse
import sys

import chess

from .capture import get_capturer
from .config import Settings, load_settings
from .engine.uci import Book, EngineClient, find_engine, parse_tc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chessbot", description="Screen-reading chess bot")
    sub = parser.add_subparsers(dest="command", required=True)

    play = sub.add_parser("play", help="detect the on-screen board and play")
    play.add_argument("--engine", help="path to a UCI engine (default: stockfish on PATH)")
    play.add_argument("--turn", choices=["white", "black"], help="side to move in the detected position")
    play.add_argument("--capture", choices=["auto", "mss", "dxcam"], default="auto")
    play.add_argument("--config", default="default.ini")
    play.add_argument("--model", help="path to the piece classifier ONNX model")
    play.add_argument("--tc", help='classical control "40/Xu" (u=ms|s|m|h, default minutes)')
    play.add_argument("--depth-mode", action="store_true")
    play.add_argument("--depth", type=int)
    play.add_argument("--threads", type=int)
    play.add_argument("--hash", type=int)
    play.add_argument("--timing-mode", choices=["delay", "engine", "both"])
    play.add_argument("--timing-min", type=float)
    play.add_argument("--timing-max", type=float)
    play.add_argument("--remote", help="host:port of a chessbot server to use instead of a local engine")

    server = sub.add_parser("server", help="serve engine moves over a socket")
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=6751)
    server.add_argument("--engine")
    server.add_argument("--threads", type=int)
    server.add_argument("--hash", type=int, default=1024)
    return parser


def resolve_timing(args) -> tuple[str | None, tuple[float, float] | None]:
    if not args.timing_mode:
        return None, None
    if args.timing_min is None or args.timing_max is None:
        raise SystemExit("--timing-mode requires both --timing-min and --timing-max")
    if args.timing_min < 0 or args.timing_max < 0:
        raise SystemExit("timing bounds must be non-negative")
    if args.timing_min > args.timing_max:
        raise SystemExit("--timing-min must be <= --timing-max")
    return args.timing_mode, (float(args.timing_min), float(args.timing_max))


def _apply_overrides(settings: Settings, args) -> Settings:
    if args.threads is not None:
        settings.threads = max(1, args.threads)
    if args.hash is not None:
        settings.hash_mb = max(16, args.hash)
    if getattr(args, "depth", None) is not None:
        settings.depth = max(1, args.depth)
    return settings


def cmd_play(args) -> None:
    from .control.mouse import Mouse
    from .game import GameSession
    from .vision.recognizer import Recognizer

    settings = _apply_overrides(load_settings(args.config), args)
    if args.tc:
        settings.move_time = parse_tc(args.tc)
        print(f"Classical control: {settings.move_time:.3f}s per move")
    timing_mode, timing_window = resolve_timing(args)

    model_path = args.model or settings.model_path
    capturer = get_capturer(args.capture)
    if args.remote:
        from .engine.remote import RemoteEngine

        host, _, port = args.remote.partition(":")
        engine = RemoteEngine(host, int(port or 6751))
    else:
        engine = EngineClient(find_engine(args.engine), settings.threads, settings.hash_mb)
    turn_arg = None
    if args.turn:
        turn_arg = chess.WHITE if args.turn == "white" else chess.BLACK

    session = GameSession(
        capturer=capturer,
        engine=engine,
        book=Book(settings.book_path),
        recognizer=Recognizer(model_path=model_path),
        mouse=Mouse(scale=capturer.scale),
        depth_mode=args.depth_mode,
        depth=settings.depth,
        move_time=settings.move_time,
        timing_mode=timing_mode,
        timing_window=timing_window,
        turn_arg=turn_arg,
    )
    try:
        session.run()
    finally:
        engine.close()
        capturer.close()


def cmd_server(args) -> None:
    from .engine.server import serve

    threads = args.threads or max(1, (__import__("os").cpu_count() or 2) - 1)
    engine = EngineClient(find_engine(args.engine), threads, args.hash)
    try:
        serve(engine, args.host, args.port)
    finally:
        engine.close()


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "play":
            cmd_play(args)
        else:
            cmd_server(args)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))
    except KeyboardInterrupt:
        sys.exit("\ninterrupted")
