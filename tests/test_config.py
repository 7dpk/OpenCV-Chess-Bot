from chessbot.config import Settings, load_settings


def test_defaults_when_file_missing(tmp_path):
    settings = load_settings(tmp_path / "nope.ini")
    assert settings == Settings()


def test_loads_legacy_ini(tmp_path):
    ini = tmp_path / "default.ini"
    ini.write_text(
        "[Engine Default Settings]\nthread=2\nhash=256\n\n[Time Control]\ndepth=15\ntime=1.5\n"
    )
    settings = load_settings(ini)
    assert settings.threads == 2
    assert settings.hash_mb == 256
    assert settings.depth == 15
    assert settings.move_time == 1.5


def test_partial_ini_keeps_defaults(tmp_path):
    ini = tmp_path / "default.ini"
    ini.write_text("[Engine Default Settings]\nthread=8\n")
    settings = load_settings(ini)
    assert settings.threads == 8
    assert settings.hash_mb == Settings().hash_mb


def test_paths_exist_in_repo():
    assert Settings().book_path.exists()
