import random
import time
import os
from typing import Set, Tuple

import cv2
import numpy as np
import pyautogui
from mss import mss

from scipy.signal import convolve2d

THRESHOLD = 10


def get_target_coords(target, array):
    res = cv2.matchTemplate(
        array.astype(np.uint8), target.astype(np.uint8), cv2.TM_CCOEFF_NORMED
    )
    _, max_amt, _, max_loc = cv2.minMaxLoc(res)
    return max_loc[::-1] if max_amt > 0.999 else None


"""
kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

convolve2d(base_array, kernel)[1:-1, 1:-1]


If nothing else, click a random unrevealed tile


Lowest hanging fruit: click all unrevealed tiles next to neighbors <= level


probably not this, not analogous to human heuristic:
for point in grid:
    if revealed (0 value or monster):
        for unrevealed neighbor:
        
"""


# Dict which accepts np bool arrays as keys
class NumpyDict:
    def __init__(self):
        self.core_dict = {}

    def __setitem__(self, key, value):
        parsed_key = np.packbits(key).tobytes()
        self.core_dict[parsed_key] = value

    def __getitem__(self, item):
        parsed_key = np.packbits(item).tobytes()
        return self.core_dict[parsed_key]


class GameState:
    def __init__(self):
        self.templates = {}
        self.populate_templates()

        self.lookup_dict = NumpyDict()
        self.populate_lookup_dict()

        self.sct = mss()

        self.top_corner = self.bottom_corner = (0, 1000)
        self.screen = self.level = self.hit_points = None
        self.update_game_state(initialize=True)
        self.grid_top_left, self.grid_bottom_right, self.grid_width = self.map_grid()
        height = (self.grid_bottom_right[0] - self.grid_top_left[0]) // self.grid_width
        width = (self.grid_bottom_right[1] - self.grid_top_left[1]) // self.grid_width
        self.grid_values = np.zeros((height, width), dtype=np.uint8)
        self.neighbor_count = np.zeros((height, width), dtype=np.uint8)
        self.is_revealed = np.zeros((height, width), dtype=bool)
        self.monster_counts = self.get_monster_counts()

    def populate_templates(self):
        for filename in os.listdir("templates"):
            match filename:
                case "colon.bmp":
                    key = ":"
                case "space.bmp":
                    key = " "
                case _:
                    key = filename.split(".")[0]
            self.templates[key] = (
                cv2.imread(f"templates/{filename}", cv2.IMREAD_GRAYSCALE) > THRESHOLD
            )

    def populate_lookup_dict(self):
        # Unchecked (all bright) and empty (all dark) grids should read as 0
        char_array = np.ones((7, 12), dtype=bool)
        self.lookup_dict[char_array] = (0, 0)
        char_array = np.zeros((7, 12), dtype=bool)
        self.lookup_dict[char_array] = (0, 0)

        # "# " should return the number
        for i in range(10):
            char_array = self.str_to_array(f"{i} ")
            self.lookup_dict[char_array] = (i, 0)

        # Number combos incl. in center of patch
        for i in range(73):
            char_array = self.str_to_array(f"{i}")
            if i < 10:
                base = np.zeros((7, 12), dtype=bool)
                base[:, 3:9] = char_array
                char_array = base
            self.lookup_dict[char_array] = (i, 0)

        # Monsters
        for i in range(1, 10):
            char_array = self.templates[f"mon_{i}"]
            self.lookup_dict[char_array] = (0, i)

    def find_game_screen_coordinates(self, screen):
        top_corner = get_target_coords(self.templates["LV"], screen)

        pointer = [top_corner[0] + 100, top_corner[1]]
        while np.any(screen[pointer[0] - 3 : pointer[0], pointer[1]]) or not np.all(
            screen[pointer[0] : pointer[0] + 3, pointer[1]]
        ):
            pointer[0] += 1
        pointer[0] -= 1
        while not screen[pointer[0], pointer[1]]:
            pointer[1] += 1
        bottom_corner = (pointer[0] + 1, pointer[1])
        return top_corner, bottom_corner

    def update_game_state(self, initialize=False):
        pyautogui.moveTo(self.bottom_corner[1] // 2, self.bottom_corner[0] // 2)

        img = (
            cv2.cvtColor(
                np.array(self.sct.grab(self.sct.monitors[1])), cv2.COLOR_BGRA2GRAY
            )
            > THRESHOLD
        )
        if initialize:
            self.top_corner, self.bottom_corner = self.find_game_screen_coordinates(img)

        img = img[
            self.top_corner[0] : self.bottom_corner[0],
            self.top_corner[1] : self.bottom_corner[1],
        ]

        self.screen = img[::2, ::2]
        self.level = self.get_following_value(3, self.screen[::2, ::2])
        self.hit_points = self.get_following_value(8, self.screen[::2, ::2])

        if initialize or self.hit_points == 0:
            return

        self.screen = self.screen[
            self.grid_top_left[0] : self.grid_bottom_right[0],
            self.grid_top_left[1] : self.grid_bottom_right[1],
        ]

        for (x, y), value in np.ndenumerate(self.is_revealed):
            number_array = self.screen[
                16 * x + 4 : 16 * x + 11, 16 * y + 2 : 16 * y + 14
            ]
            neighbors, own_count = self.lookup_dict[number_array]
            self.neighbor_count[x, y] = neighbors
            self.is_revealed[x, y] = 0 in number_array
            if own_count:
                # Found a monster, register its value and click it to get the neighbors later
                self.monster_counts[own_count] -= 1
                self.grid_values[x, y] = own_count
                self.click_grid(x, y)

        self.monster_counts[0] = (
            len(self.is_revealed.flatten()) - self.monster_counts[1:].sum()
        )

    def get_monster_counts(self):
        count_array = np.zeros(10, dtype=np.uint8)
        for i in range(1, 10):
            count_target = self.str_to_array(f"LV{i}:x")
            count_coords = get_target_coords(count_target, self.screen)
            if count_coords:
                monster_count = self.get_following_value(5, self.screen, count_coords)
                count_array[i] = monster_count

        count_array[0] = len(self.is_revealed.flatten()) - count_array.sum()

        return count_array

    def str_to_array(self, to_convert):
        vert_break = np.zeros((7, 1), dtype=bool)
        char_arrays = []
        for character in to_convert:
            char_arrays.append(self.templates[character])
            char_arrays.append(vert_break)

        return np.hstack(char_arrays)

    def get_following_value(self, target_chars, array, coords=(0, 0)):
        arr = array[
            coords[0] : coords[0] + 7,
            coords[1] + 6 * target_chars : coords[1] + 6 * target_chars + 12,
        ]
        return self.lookup_dict[arr][0]

    def map_grid(self):
        pointer = [0, 0]
        while self.screen[pointer[0], pointer[1]]:
            pointer[0] += 1
        pointer[0] += 1
        while self.screen[pointer[0], pointer[1]] == 0:
            pointer[1] += 1
        top_left = pointer.copy()
        # while self.screen[pointer[0], pointer[1]]:
        #     pointer[0] += 1
        # width = pointer[0] - top_left[0] + 1
        width = 16
        while (
            self.screen[pointer[0] + 1, pointer[1]]
            or self.screen[pointer[0] + 2, pointer[1]]
        ):
            pointer[0] += 1
        while (
            self.screen[pointer[0], pointer[1]]
            or self.screen[pointer[0], pointer[1] + 2]
        ):
            pointer[1] += 1
        bottom_right = [pointer[0] + 2, pointer[1] + 1]
        return top_left, bottom_right, width

    def click_grid(self, x, y):
        # Debugging to get game in focus, remove later
        pyautogui.click(
            clicks=1,
            x=self.top_corner[1] // 2
            + self.grid_top_left[1]
            + self.grid_width * -1
            + self.grid_width // 2,
            y=self.top_corner[0] // 2
            + self.grid_top_left[0]
            + self.grid_width * -1
            + self.grid_width // 2,
        )
        pyautogui.click(
            clicks=1,
            x=self.top_corner[1] // 2
            + self.grid_top_left[1]
            + self.grid_width * y
            + self.grid_width // 2,
            y=self.top_corner[0] // 2
            + self.grid_top_left[0]
            + self.grid_width * x
            + self.grid_width // 2,
        )

    def get_random_unrevealed(self) -> Set[Tuple[int, int]]:
        x, y = np.where(~self.is_revealed)
        point = random.randrange(len(x))
        return {(x[point], y[point])}

    def get_unrevealed_under_level(self) -> Set[Tuple[int, int]]:
        pass


def solve(game: GameState):
    to_click = {game.get_random_unrevealed()}
    game.click_grid(-1, -1)
    while True:
        while to_click:
            game.click_grid(*to_click.pop())

        game.update_game_state()

        if game.hit_points == 0:
            return

        # Try to identify clickable squares.
        # Start simple, get difficult, end with a random guess (weighted?)
        to_click = game.get_unrevealed_under_level() or game.get_random_unrevealed()


def main():
    game = GameState()
    while True:
        game.click_grid(-1, -1)
        solve(game)
        if input("Continue? y/n") != "y":
            break


if __name__ == "__main__":
    main()
