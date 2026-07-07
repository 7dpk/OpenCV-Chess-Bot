import configparser
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    threads: int = 4
    hash_mb: int = 512
    depth: int = 12
    move_time: float = 2.0
    book_path: Path = field(default_factory=lambda: REPO_ROOT / "assets" / "Performance.bin")
    model_path: Path = field(default_factory=lambda: REPO_ROOT / "models" / "piece_classifier.onnx")


def load_settings(ini_path: str | Path = "default.ini") -> Settings:
    settings = Settings()
    parser = configparser.ConfigParser()
    if not parser.read(ini_path):
        return settings
    engine = parser["Engine Default Settings"] if parser.has_section("Engine Default Settings") else {}
    tc = parser["Time Control"] if parser.has_section("Time Control") else {}
    settings.threads = int(engine.get("thread", settings.threads))
    settings.hash_mb = int(engine.get("hash", settings.hash_mb))
    settings.depth = int(tc.get("depth", settings.depth))
    settings.move_time = float(tc.get("time", settings.move_time))
    return settings
