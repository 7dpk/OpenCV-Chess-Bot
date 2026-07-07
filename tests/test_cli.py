import pytest

import chessbot.cli as cli
from chessbot.cli import build_parser, main, resolve_timing


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


def test_play_missing_model_exits_before_engine_spawn(monkeypatch):
    capturer_calls = []
    engine_calls = []

    def fake_get_capturer(backend):
        capturer_calls.append(backend)
        raise AssertionError("get_capturer should not be called when the model is missing")

    class FakeEngineClient:
        def __init__(self, *args, **kwargs):
            engine_calls.append((args, kwargs))
            raise AssertionError("EngineClient should not be instantiated when the model is missing")

    monkeypatch.setattr(cli, "get_capturer", fake_get_capturer)
    monkeypatch.setattr(cli, "EngineClient", FakeEngineClient)

    with pytest.raises(SystemExit) as excinfo:
        main(["play", "--model", "/nonexistent/model.onnx"])

    assert "/nonexistent/model.onnx" in str(excinfo.value)
    assert engine_calls == []
    assert capturer_calls == []


def test_main_reports_os_error_as_system_exit(monkeypatch):
    def fake_cmd_play(args):
        raise ConnectionRefusedError("connection refused to 1.2.3.4:6751")

    monkeypatch.setattr(cli, "cmd_play", fake_cmd_play)

    with pytest.raises(SystemExit) as excinfo:
        main(["play"])

    assert str(excinfo.value) == "connection refused to 1.2.3.4:6751"
