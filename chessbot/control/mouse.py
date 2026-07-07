import time

from ..vision.squares import square_center


class Mouse:
    def __init__(self, scale: float, gui=None, promotion_delay: float = 0.2):
        if gui is None:
            import pyautogui

            pyautogui.PAUSE = 0.01
            gui = pyautogui
        self._gui = gui
        self._scale = scale
        self._promotion_delay = promotion_delay

    def click_pixel(self, x: float, y: float) -> None:
        self._gui.click(x / self._scale, y / self._scale)

    def click_square(self, name: str, white_at_bottom: bool, region: tuple) -> None:
        x, y = square_center(name, white_at_bottom, region)
        self.click_pixel(x, y)

    def play_move(self, uci: str, white_at_bottom: bool, region: tuple) -> None:
        self.click_square(uci[:2], white_at_bottom, region)
        self.click_square(uci[2:4], white_at_bottom, region)
        if len(uci) == 5:
            time.sleep(self._promotion_delay)
            self.click_square(uci[2:4], white_at_bottom, region)
