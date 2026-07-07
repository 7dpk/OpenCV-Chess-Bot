import sys

import numpy as np
import pytest

from chessbot.capture import get_capturer


def _capturer_or_skip():
    try:
        cap = get_capturer("mss")
        img = cap.grab()
    except Exception as exc:  # no display / no screen-recording permission
        pytest.skip(f"screen capture unavailable: {exc}")
    return cap, img


def test_mss_full_grab_shape():
    cap, img = _capturer_or_skip()
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[0] > 100 and img.shape[1] > 100
    cap.close()


def test_mss_region_grab():
    cap, full = _capturer_or_skip()
    region = (0, 0, 128, 128)
    img = cap.grab(region)
    assert img.shape[2] == 3
    assert abs(img.shape[0] - 128) <= 4 and abs(img.shape[1] - 128) <= 4
    cap.close()


def test_mss_scale_positive():
    cap, _ = _capturer_or_skip()
    assert cap.scale >= 1.0
    cap.close()


def test_factory_auto_selects_platform_backend():
    cap, _ = _capturer_or_skip()
    cap.close()
    auto = get_capturer("auto")
    if sys.platform == "win32":
        assert auto.name in ("dxcam", "mss")
    else:
        assert auto.name == "mss"
    auto.close()


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError):
        get_capturer("nope")
