import random
import time
from typing import Set, Tuple, Optional

import cv2
import numpy as np
import pyautogui
from mss import mss

from utils import THRESHOLD, templates, get_target_coords, str_to_array, lookup_dict

from scipy.signal import convolve2d

pyautogui.PAUSE = 0


class NoHitPoints(Exception):
    pass


class Win(Exception):
    pass


class GameState:
    def __init__(self):
        self.sct = mss()

        self.top_corner = self.bottom_corner = (0, 1000)
        self.screen = np.array([1])
        self.level = self.hit_points = None
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
            else:
                time.sleep(0.01)
                print("screen shaking")

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

    def click_grid(self, x, y):
        # Debugging to get game in focus, remove later
        # pyautogui.click(
        #     x=self.top_corner[1] // 2 + self.grid_top_left[1] + self.grid_width // 2,
        #     y=self.top_corner[0] // 2
        #     + self.grid_top_left[0]
        #     + self.grid_width * -1
        #     + self.grid_width // 2,
        # )
        pyautogui.click(
            x=self.top_corner[1] // 2
            + self.grid_top_left[1]
            + self.grid_width * y
            + self.grid_width // 2,
            y=self.top_corner[0] // 2
            + self.grid_top_left[0]
            + self.grid_width * x
            + self.grid_width // 2,
        )
        if x >= 0 and y >= 0:
            self.unrevealed[x, y] = False
            if np.all(~self.unrevealed):
                raise Win("You win!")

    def get_random_unrevealed(self) -> Set[Tuple[int, int]]:
        x, y = np.where(
            self.unrevealed
        )  # TODO: make sure you don't pick a known high value
        point = random.randrange(len(x))
        return {(x[point], y[point])}

    def get_unrevealed_under_level(self) -> Set[Tuple[int, int]]:
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        # unrevealed_neighbors is 8 if all neighbors are unrevealed (boring) or 0 if all are revealed (boring)
        unrevealed_neighbors = convolve2d(self.unrevealed, kernel, boundary="symm")[
            1:-1, 1:-1
        ]
        # # only clickable points for this heuristic
        # unrevealed_boundary = np.all((unrevealed_neighbors < 8, self.unrevealed), axis=0)
        # must have a neighbor with a number
        number_boundary = np.all((unrevealed_neighbors > 0, ~self.unrevealed), axis=0)
        contacting_nums_under_level = np.all(
            (number_boundary, self.neighbor_count <= self.level), axis=0
        )
        neighbor_under_level = (
            convolve2d(contacting_nums_under_level, kernel)[1:-1, 1:-1] > 0
        )
        safe_to_click = np.all((neighbor_under_level, self.unrevealed), axis=0)
        x, y = np.where(safe_to_click)
        return set(zip(x, y))


def solve(game: GameState):
    game.click_grid(0, -1)
    game.click_grid(0, -1)
    to_click = game.get_random_unrevealed()
    while True:
        try:
            while to_click:
                game.click_grid(*to_click.pop())

            # Start simple, get difficult, end with a random guess (weighted?)
            to_click = (
                game.update_game_state()
                or game.update_game_state()
                or game.get_unrevealed_under_level()
                or game.get_random_unrevealed()
            )
            a = 1
        except Exception as e:
            # cv2.imwrite("fail.png", game.screen.astype(int) * 255)
            print(type(e))
            raise e


def main():
    game = GameState()
    while True:
        try:
            solve(game)
        except Exception as e:
            print(e)
            if input("Continue? y/n") == "y":
                game.click_grid(0, -1)
                game.click_grid(0, -1)
                game.__init__()
            else:
                break


if __name__ == "__main__":
    main()
