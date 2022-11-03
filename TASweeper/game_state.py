import cv2
import numpy as np
import pyautogui
from mss import mss
from scipy.signal import convolve2d

from utils import (
    THRESHOLD,
    templates,
    get_target_coords,
    str_to_array,
    lookup_dict,
    get_following_value,
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

        self.top_corner = self.bottom_corner = (0, 1000)
        self.screen = self.level = self.hit_points = None
        self.update_game_state(initialize=True)
        self.grid_top_left, self.grid_bottom_right, self.grid_width = self.map_grid()
        height = (self.grid_bottom_right[0] - self.grid_top_left[0]) // self.grid_width
        width = (self.grid_bottom_right[1] - self.grid_top_left[1]) // self.grid_width
        self.grid_values = np.zeros((height, width), dtype=np.uint8)
        self.grid_value_known = np.zeros((height, width), dtype=bool)
        self.neighbor_count = np.zeros((height, width), dtype=np.uint8)
        self.neighbor_known = np.zeros((height, width), dtype=bool)
        self.unrevealed = np.ones((height, width), dtype=bool)
        self._monster_counts = self.get_monster_counts()

    @property
    def monster_counts(self):
        # Refactor to use grid_values (if needed)
        raise NotImplementedError()

    def copy_state(self):
        return {
            "unrevealed": np.copy(self.unrevealed),
            "grid_values": np.copy(self.grid_values),
            "grid_value_known": np.copy(self.grid_value_known),
            "neighbor_count": np.copy(self.neighbor_count),
            "modified_neighbor_count": np.copy(self.modified_neighbor_count),
            "neighbor_known": np.copy(self.neighbor_known),
        }

    def find_game_screen_coordinates(self):
        top_corner = get_target_coords(templates["LV"], self.screen)

        pointer = [top_corner[0] + 100, top_corner[1]]
        while np.any(
            self.screen[pointer[0] - 3 : pointer[0], pointer[1]]
        ) or not np.all(self.screen[pointer[0] : pointer[0] + 3, pointer[1]]):
            pointer[0] += 1
        pointer[0] -= 1
        while not self.screen[pointer[0], pointer[1]]:
            pointer[1] += 1
        bottom_corner = (pointer[0] + 1, pointer[1])
        return top_corner, bottom_corner

    def get_monster_counts(self):
        count_array = np.zeros(10, dtype=np.uint8)
        for i in range(1, 10):
            count_target = str_to_array(f"LV{i}:x")
            count_coords = get_target_coords(count_target, self.screen)
            if count_coords:
                monster_count = get_following_value(5, self.screen, count_coords)
                count_array[i] = monster_count

        count_array[0] = len(self.unrevealed.flatten()) - count_array.sum()

        return count_array

    def map_grid(self):
        pointer = [0, 0]
        while self.screen[pointer[0], pointer[1]]:
            pointer[0] += 1
        pointer[0] += 1
        while self.screen[pointer[0], pointer[1]] == 0:
            pointer[1] += 1
        top_left = pointer.copy()
        while (
            self.screen[pointer[0] + 1, pointer[1]]
            or self.screen[pointer[0] + 2, pointer[1]]
        ):
            pointer[0] += 1
        while pointer[1] + 2 < self.screen.shape[1] and (
            self.screen[pointer[0], pointer[1]]
            or self.screen[pointer[0], pointer[1] + 2]
        ):
            pointer[1] += 1
        if pointer[1] + 2 >= self.screen.shape[1]:
            pointer[1] += 1
        bottom_right = [pointer[0] + 2, pointer[1] + 1]
        return top_left, bottom_right, 16

    def click_grid(self, x, y, clicks=1):
        pyautogui.click(
            x=self.top_corner[1] // 2
            + self.grid_top_left[1]
            + self.grid_width * y
            + self.grid_width // 2,
            y=self.top_corner[0] // 2
            + self.grid_top_left[0]
            + self.grid_width * x
            + self.grid_width // 2,
            clicks=clicks,
        )
        if x >= 0 and y >= 0:
            self.unrevealed[x, y] = False
            if np.all(~self.unrevealed):
                raise Win("You win!")

    def update_game_state(self, initialize=False):
        pyautogui.moveTo(self.bottom_corner[1] // 2, self.bottom_corner[0] // 2)

        while True:
            # noinspection PyTypeChecker
            self.screen = (
                cv2.cvtColor(
                    np.array(self.sct.grab(self.sct.monitors[1])), cv2.COLOR_BGRA2GRAY
                )
                > THRESHOLD
            )

            if initialize:
                (
                    self.top_corner,
                    self.bottom_corner,
                ) = self.find_game_screen_coordinates()

            self.screen = self.screen[
                self.top_corner[0] : self.bottom_corner[0],
                self.top_corner[1] : self.bottom_corner[1],
            ]

            if np.all(
                np.equal(
                    self.screen[: templates["LV"].shape[0], : templates["LV"].shape[1]],
                    templates["LV"],
                )
            ):
                break

        self.screen = self.screen[::2, ::2]
        self.level = get_following_value(3, self.screen[::2, ::2])
        self.hit_points = get_following_value(8, self.screen[::2, ::2])

        if initialize:
            return

        if self.hit_points == 0:
            raise NoHitPoints("You Died - RIP")

        self.screen = self.screen[
            self.grid_top_left[0] : self.grid_bottom_right[0],
            self.grid_top_left[1] : self.grid_bottom_right[1],
        ]
        if not self.screen[0, 0] or np.all(~self.unrevealed):
            raise Win(f"Win: {not self.screen[0, 0]}, {np.all(~self.unrevealed)}")

        for (x, y), _ in np.ndenumerate(self.unrevealed):
            number_array = self.screen[
                16 * x + 4 : 16 * x + 11, 16 * y + 2 : 16 * y + 14
            ]
            if ~np.all(number_array):  # Filled grids need no action
                neighbors, own_count = lookup_dict[number_array]
                self.unrevealed[x, y] = False
                if own_count:  # Visible monster
                    self.grid_values[x, y] = own_count
                    self.grid_value_known[x, y] = True
                else:  # Number, clicked monster, or blank
                    self.neighbor_count[x, y] = neighbors
                    self.grid_value_known[x, y] = True
                    self.neighbor_known[x, y] = True
