from typing import Optional, Protocol, Tuple

import numpy as np

Region = Tuple[int, int, int, int]


class Capturer(Protocol):
    name: str

    @property
    def scale(self) -> float: ...

    def grab(self, region: Optional[Region] = None) -> np.ndarray: ...

    def close(self) -> None: ...
