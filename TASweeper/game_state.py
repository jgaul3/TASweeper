from typing import Set, Tuple, Optional

import cv2
import numpy as np
import pyautogui
from mss import mss

from utils import THRESHOLD, templates, get_target_coords, str_to_array, lookup_dict


class NoHitPoints(Exception):
    pass


class Win(Exception):
    pass


class GameState:
    def __init__(self):
        self.sct = mss()

        self.top_corner = self.bottom_corner = (0, 1000)
        self.screen = self.level = self.hit_points = None
        self.update_game_state(initialize=True)
        self.grid_top_left, self.grid_bottom_right, self.grid_width = self.map_grid()
        height = (self.grid_bottom_right[0] - self.grid_top_left[0]) // self.grid_width
        width = (self.grid_bottom_right[1] - self.grid_top_left[1]) // self.grid_width
        self.grid_values = np.zeros((height, width), dtype=np.uint8)
        self.neighbor_count = np.zeros((height, width), dtype=np.uint8)
        self.unrevealed = np.ones((height, width), dtype=bool)
        self.monster_counts = self.get_monster_counts()

    def update_game_state(self, initialize=False) -> Optional[Set[Tuple[int, int]]]:
        pyautogui.moveTo(self.bottom_corner[1] // 2, self.bottom_corner[0] // 2)

        while True:
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
        self.level = self.get_following_value(3, self.screen[::2, ::2])
        self.hit_points = self.get_following_value(8, self.screen[::2, ::2])

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

        to_click = set()
        for (x, y), _ in np.ndenumerate(self.unrevealed):
            number_array = self.screen[
                16 * x + 4 : 16 * x + 11, 16 * y + 2 : 16 * y + 14
            ]
            neighbors, own_count = lookup_dict[number_array]
            self.unrevealed[x, y] = np.all(number_array)
            if own_count:
                # Found a monster, register its value and click it to get the neighbors later
                self.monster_counts[own_count] -= 1
                self.grid_values[x, y] = own_count
                to_click.add((x, y))
            else:
                self.neighbor_count[x, y] = neighbors

        self.monster_counts[0] = (
            len(self.unrevealed.flatten()) - self.monster_counts[1:].sum()
        )

        return to_click

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
                monster_count = self.get_following_value(5, self.screen, count_coords)
                count_array[i] = monster_count

        count_array[0] = len(self.unrevealed.flatten()) - count_array.sum()

        return count_array

    def get_following_value(self, target_chars, array, coords=(0, 0)):
        arr = array[
            coords[0] : coords[0] + 7,
            coords[1] + 6 * target_chars : coords[1] + 6 * target_chars + 12,
        ]
        return lookup_dict[arr][0]

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
