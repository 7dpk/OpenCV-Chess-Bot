import time

import numpy as np

from .base import Region


class DxcamCapturer:
    """Windows Desktop Duplication capture. dxcam returns BGR when configured so,
    and returns None from grab() when nothing changed; we cache the last frame."""

    name = "dxcam"

    def __init__(self):
        import dxcam

        self._cam = dxcam.create(output_color="BGR")
        if self._cam is None:
            raise RuntimeError("dxcam could not create a camera")
        self._last: np.ndarray | None = None

    @property
    def scale(self) -> float:
        return 1.0

    def grab(self, region: Region | None = None) -> np.ndarray:
        frame = self._cam.grab()
        if frame is not None:
            self._last = frame
        elif self._last is None:
            deadline = time.time() + 2.0
            while self._last is None and time.time() < deadline:
                frame = self._cam.grab()
                if frame is not None:
                    self._last = frame
                else:
                    time.sleep(0.005)
            if self._last is None:
                raise RuntimeError("dxcam produced no frames")
        img = self._last
        if region is not None:
            left, top, right, bottom = region
            img = img[top:bottom, left:right]
        return np.ascontiguousarray(img)

    def close(self) -> None:
        self._cam.release()
