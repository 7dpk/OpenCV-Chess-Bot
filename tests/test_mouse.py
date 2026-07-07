from chessbot.control.mouse import Mouse


class FakeGui:
    def __init__(self):
        self.clicks = []

    def click(self, x, y):
        self.clicks.append((x, y))


def test_click_pixel_applies_scale():
    gui = FakeGui()
    mouse = Mouse(scale=2.0, gui=gui)
    mouse.click_pixel(100, 300)
    assert gui.clicks == [(50.0, 150.0)]


def test_play_move_clicks_centers():
    gui = FakeGui()
    mouse = Mouse(scale=1.0, gui=gui)
    mouse.play_move("e2e4", True, (0, 0, 800, 800))
    assert gui.clicks == [(450.0, 650.0), (450.0, 450.0)]


def test_play_move_promotion_clicks_end_square_again():
    gui = FakeGui()
    mouse = Mouse(scale=1.0, gui=gui, promotion_delay=0.0)
    mouse.play_move("e7e8q", True, (0, 0, 800, 800))
    assert len(gui.clicks) == 3
    assert gui.clicks[1] == gui.clicks[2]
