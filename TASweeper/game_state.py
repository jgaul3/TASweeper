import time

import cv2
import numpy as np
from mss import mss
from pynput.mouse import Controller, Button, Listener
from scipy.signal import convolve2d

from utils import (
    THRESHOLD,
    templates,
    get_target_coords,
    lookup_dict,
)


class NoHitPoints(Exception):
    pass


class Win(Exception):
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
        self.click_grid(-1, 0, 2)

    @property
    def modified_neighbor_count(self):
        # Account for values which are high due to known monsters
        known_value_modifier = convolve2d(
            self.grid_values, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), mode="same"
        )
        return self.neighbor_count - known_value_modifier

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
            not self.screen[pointer[0], pointer[1]]  # Not off edge of screen
            and self.screen[pointer[0] - grid_height, pointer[1]]  # Under a grid square
        ):
            pointer[1] += grid_height

        # One pixel past the bottom right corner
        return top_left, pointer, grid_height, grid_scale

    def click_grid(self, x, y, clicks=2):
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

        time.sleep(0.5)

        self.mouse.click(Button.left, count=6)
        if x >= 0 and y >= 0:
            self.unrevealed[x, y] = False
            if np.all(~self.unrevealed):
                raise Win("You win!")

    def update_game_state(self, initialize=False):
        self.mouse.position = (self.bottom_corner[1] // 2, self.bottom_corner[0] // 2)
        time.sleep(0.05)

        while True:
            # noinspection PyTypeChecker
            self.screen = (
                cv2.cvtColor(
                    np.array(self.sct.grab(self.sct.monitors[1])), cv2.COLOR_BGRA2GRAY
                )
                > THRESHOLD
            )

            if initialize:
                self.find_game_screen_coordinates()

            self.screen = self.screen[
                self.top_corner[0] : self.bottom_corner[0],
                self.top_corner[1] : self.bottom_corner[1],
            ]

            # Revisit!!!
            if np.all(
                np.equal(
                    self.screen[
                        : templates["target"].shape[0], : templates["target"].shape[1]
                    ],
                    templates["target"],
                )
            ):
                break

        self.screen = self.screen[::2, ::2]
        # Get just the digits, consider it a revealed tile
        self.hit_points = lookup_dict[self.screen[4:25:3, 51:93:3]][0]
        self.level = lookup_dict[self.screen[4:25:3, 219:261:3]][0]

        if initialize:
            return

        if self.hit_points == 0:
            raise NoHitPoints("You Died - RIP")

        self.screen = self.screen[
            self.grid_top_left[0] : self.grid_bottom_right[0] : self.grid_scale,
            self.grid_top_left[1] : self.grid_bottom_right[1] : self.grid_scale,
        ]

        for (x, y), _ in np.ndenumerate(self.unrevealed):
            single_box = self.screen[
                self.scaled_grid_width * x : self.scaled_grid_width * (x + 1),
                self.scaled_grid_width * y : self.scaled_grid_width * (y + 1),
            ]
            searchable_subarray = single_box[4:11, 1:15]
            if ~np.all(searchable_subarray):  # Filled grids need no action
                neighbors, own_count = lookup_dict[searchable_subarray]
                self.unrevealed[x, y] = False
                if own_count:  # Visible monster
                    self.grid_values[x, y] = own_count
                    self.grid_value_known[x, y] = True
                else:  # Number, clicked monster, or blank
                    self.neighbor_count[x, y] = neighbors
                    self.grid_value_known[x, y] = True
                    self.neighbor_known[x, y] = True
