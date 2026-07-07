import pytest

from chessbot.cli import build_parser, resolve_timing


def test_play_defaults():
    args = build_parser().parse_args(["play"])
    assert args.command == "play"
    assert args.capture == "auto"
    assert args.turn is None


def test_play_all_flags():
    args = build_parser().parse_args(
        [
            "play", "--engine", "/usr/bin/stockfish", "--turn", "black", "--capture", "mss",
            "--tc", "40/5m", "--depth-mode", "--depth", "14", "--threads", "2", "--hash", "256",
            "--timing-mode", "both", "--timing-min", "1", "--timing-max", "3",
            "--remote", "1.2.3.4:6751", "--model", "models/piece_classifier.onnx",
        ]
    )
    assert args.turn == "black" and args.depth == 14 and args.remote == "1.2.3.4:6751"


def test_server_flags():
    args = build_parser().parse_args(["server", "--port", "7000", "--hash", "2048"])
    assert args.command == "server" and args.port == 7000


def test_resolve_timing_valid():
    args = build_parser().parse_args(
        ["play", "--timing-mode", "delay", "--timing-min", "1", "--timing-max", "2"]
    )
    assert resolve_timing(args) == ("delay", (1.0, 2.0))


def test_resolve_timing_none():
    args = build_parser().parse_args(["play"])
    assert resolve_timing(args) == (None, None)


@pytest.mark.parametrize(
    "flags",
    [
        ["--timing-mode", "delay"],
        ["--timing-mode", "delay", "--timing-min", "2", "--timing-max", "1"],
        ["--timing-mode", "delay", "--timing-min", "-1", "--timing-max", "1"],
    ],
)
def test_resolve_timing_invalid(flags):
    args = build_parser().parse_args(["play", *flags])
    with pytest.raises(SystemExit):
        resolve_timing(args)
