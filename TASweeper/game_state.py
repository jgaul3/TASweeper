import time

import cv2
import numpy as np
from mss import mss
from pynput.mouse import Controller, Button

from utils import (
    HIGH_THRESHOLD,
    templates,
    get_target_coords,
    lookup_dict,
    THRESHOLD,
)


class NoHitPoints(Exception):
    pass


class Win(Exception):
    pass


class RedoUpdate(Exception):
    pass


class GameState:
    """
    GameState holds all information related to the game's current state.

    All calculations should be done elsewhere.
    """

    def __init__(self):
        self.sct = mss()
        self.mouse = Controller()

        self.top_corner = self.bottom_corner = (0, 1000)
        self.screen = self.level = self.hit_points = None
        self.update_game_state(initialize=True)
        (
            self.grid_top_left,
            self.grid_bottom_right,
            self.grid_width,
            self.grid_scale,
        ) = self.map_grid()
        self.scaled_grid_width = self.grid_width // self.grid_scale
        height = (self.grid_bottom_right[0] - self.grid_top_left[0]) // self.grid_width
        width = (self.grid_bottom_right[1] - self.grid_top_left[1]) // self.grid_width
        self.grid_values = np.zeros((height, width), dtype=np.uint8)
        self.grid_value_known = np.zeros((height, width), dtype=bool)
        self.neighbor_count = np.zeros((height, width), dtype=np.uint8)
        self.neighbor_known = np.zeros((height, width), dtype=bool)
        self.unrevealed = np.ones((height, width), dtype=bool)
        self.clicked_squares = set()
        self.click_grid(-1, 0)
        self.click_grid(-1, 0)

    def copy_state(self):
        return {
            "unrevealed": np.copy(self.unrevealed),
            "grid_values": np.copy(self.grid_values),
            "grid_value_known": np.copy(self.grid_value_known),
            "neighbor_count": np.copy(self.neighbor_count),
            "neighbor_known": np.copy(self.neighbor_known),
        }

    def find_game_screen_coordinates(self):
        self.top_corner = get_target_coords(templates["target"], self.screen)

        pointer = [self.top_corner[0], self.top_corner[1]]
        while not (
            np.any(self.screen[pointer[0] - 100 : pointer[0], pointer[1]])
            and np.all(self.screen[pointer[0] : pointer[0] + 100, pointer[1]])
        ):
            pointer[0] += 1
        while not self.screen[pointer[0] - 2, pointer[1]]:
            pointer[1] += 1
        self.bottom_corner = (pointer[0], pointer[1])

    def map_grid(self):
        pointer = [64, 0]
        grid_height, grid_scale = 0, 0

        # Get top_left corner
        while not self.screen[pointer[0], pointer[1]]:
            pointer[1] += 1
            if pointer[1] == self.screen.shape[1]:
                pointer = [pointer[0] + 1, 0]
        top_left = pointer.copy()

        while self.screen[pointer[0], pointer[1]]:
            pointer[0] += 1
            grid_height += 1

        while not self.screen[pointer[0], pointer[1]]:
            pointer[0] += 1
            grid_height += 1
            grid_scale += 1

        # At top of second box down
        while self.screen[pointer[0], pointer[1]]:
            pointer[0] += grid_height

        # Just under bottom box
        while (
            pointer[1] < self.screen.shape[1]  # Not off edge of screen
            and self.screen[pointer[0] - grid_height, pointer[1]]  # Under a grid square
        ):
            pointer[1] += grid_height

        # One pixel past the bottom right corner
        return top_left, pointer, grid_height, grid_scale

    def click_grid(self, x, y):
        self.mouse.position = (
            self.top_corner[1] // 2
            + self.grid_top_left[1]
            + self.grid_width * y
            + self.grid_width // 2,
            self.top_corner[0] // 2
            + self.grid_top_left[0]
            + self.grid_width * x
            + self.grid_width // 2,
        )

        time.sleep(0.01)
        self.mouse.press(Button.left)
        time.sleep(0.03)
        self.mouse.release(Button.left)
        time.sleep(0.01)
        self.clicked_squares.add((x, y))

    def update_game_state(self, initialize=False):
        ok = False
        while not ok:
            ok = self._update_game_state(initialize)

    def _update_game_state(self, initialize) -> bool:
        # noinspection PyTypeChecker
        self.screen = cv2.cvtColor(
            np.array(self.sct.grab(self.sct.monitors[1])), cv2.COLOR_BGRA2GRAY
        )
        low_pass_screen = np.array(self.screen > THRESHOLD)
        self.screen = self.screen > HIGH_THRESHOLD

        if initialize:
            self.find_game_screen_coordinates()

        self.screen = self.screen[
            self.top_corner[0] : self.bottom_corner[0],
            self.top_corner[1] : self.bottom_corner[1],
        ]
        low_pass_screen = low_pass_screen[
            self.top_corner[0] : self.bottom_corner[0],
            self.top_corner[1] : self.bottom_corner[1],
        ]

        # Revisit!!!
        shaking = not np.all(
            np.equal(
                self.screen[
                    : templates["target"].shape[0], : templates["target"].shape[1]
                ],
                templates["target"],
            )
        )
        green_flash = np.all(self.screen[150:-150, 5])
        if shaking or green_flash:
            return False

        self.screen = self.screen[::2, ::2]
        # Get just the digits, consider it a revealed tile
        self.hit_points = lookup_dict[self.screen[4:25:3, 51:93:3]][0]
        self.level = lookup_dict[self.screen[4:25:3, 219:261:3]][0]

        if initialize:
            return True

        if self.hit_points == 0:
            raise NoHitPoints("You Died - RIP")

        low_pass_screen = low_pass_screen[::2, ::2]
        low_pass_screen = low_pass_screen[
            self.grid_top_left[0] : self.grid_bottom_right[0] : self.grid_scale,
            self.grid_top_left[1] : self.grid_bottom_right[1] : self.grid_scale,
        ]

        if not low_pass_screen[0, 0]:
            raise Win("You win!")

        for (x, y), _ in np.ndenumerate(self.unrevealed):
            single_box = low_pass_screen[
                self.scaled_grid_width * x : self.scaled_grid_width * (x + 1),
                self.scaled_grid_width * y : self.scaled_grid_width * (y + 1),
            ]
            searchable_subarray = single_box[4:11, 1:15]
            if np.all(searchable_subarray):
                # Are visuals lagging?
                if (x, y) in self.clicked_squares:
                    return False
            else:
                neighbors, own_count = lookup_dict[searchable_subarray]
                self.unrevealed[x, y] = False
                if own_count:  # Visible monster
                    self.grid_values[x, y] = own_count
                    self.grid_value_known[x, y] = True
                else:  # Number, clicked monster, or blank
                    self.neighbor_count[x, y] = neighbors
                    self.grid_value_known[x, y] = True
                    self.neighbor_known[x, y] = True
        return True
