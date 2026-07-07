import mss
import numpy as np

from .base import Region


class MssCapturer:
    name = "mss"

    def __init__(self, monitor_index: int = 1):
        self._sct = mss.mss()
        self._monitor = self._sct.monitors[monitor_index]
        self._scale: float | None = None

    @property
    def scale(self) -> float:
        if self._scale is None:
            full = self._sct.grab(self._monitor)
            self._scale = full.width / self._monitor["width"]
        return self._scale

    def grab(self, region: Region | None = None) -> np.ndarray:
        if region is None:
            box = self._monitor
        else:
            s = self.scale
            left, top, right, bottom = region
            box = {
                "left": self._monitor["left"] + round(left / s),
                "top": self._monitor["top"] + round(top / s),
                "width": max(1, round((right - left) / s)),
                "height": max(1, round((bottom - top) / s)),
            }
        shot = self._sct.grab(box)
        return np.ascontiguousarray(np.asarray(shot)[:, :, :3])

    def close(self) -> None:
        self._sct.close()
