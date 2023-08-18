import os

import cv2
import numpy as np

from TASweeper.game_state import GameState
from TASweeper.utils import THRESHOLD, templates, lookup_dict


def debug_logging(game: GameState, to_click, round_number):
    if not os.path.exists("debug_images"):
        os.makedirs("debug_images")
    to_record = game.screen.astype(int) * 127
    for x, y in to_click:
        x = game.grid_width * x + game.grid_width // 2
        y = game.grid_width * y + game.grid_width // 2
        to_record[x - 1 : x, y - 1 : y] = 255
    cv2.imwrite(f"debug_images/{round_number}.png", to_record)


def debug_clicking(game: GameState, max_hp):
    while True:
        # noinspection PyTypeChecker
        curr_screen = (
            cv2.cvtColor(
                np.array(game.sct.grab(game.sct.monitors[1])), cv2.COLOR_BGRA2GRAY
            )
            > THRESHOLD
        )

        curr_screen = curr_screen[
            game.top_corner[0] : game.bottom_corner[0],
            game.top_corner[1] : game.bottom_corner[1],
        ]

        if np.all(
            np.equal(
                curr_screen[: templates["LV"].shape[0], : templates["LV"].shape[1]],
                templates["LV"],
            )
        ):
            break

    hit_points = lookup_dict[game.screen[4:25:3, 51:93:3]][0]

    if hit_points < max_hp:
        print("huh")
