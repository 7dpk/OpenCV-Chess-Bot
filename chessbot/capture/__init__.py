import sys

from .base import Capturer, Region
from .mss_backend import MssCapturer


def get_capturer(backend: str = "auto") -> Capturer:
    if backend == "mss":
        return MssCapturer()
    if backend == "dxcam":
        from .dxcam_backend import DxcamCapturer

        return DxcamCapturer()
    if backend == "auto":
        if sys.platform == "win32":
            try:
                from .dxcam_backend import DxcamCapturer

                return DxcamCapturer()
            except Exception:
                pass
        return MssCapturer()
    raise ValueError(f"unknown capture backend: {backend!r}")


__all__ = ["Capturer", "Region", "get_capturer", "MssCapturer"]
