import math
import time
import os
import cv2
import numpy as np
import pyautogui
from mss import mss


NUM_COLORS = 16


def get_target_coords(target, array):
    res = cv2.matchTemplate(array, target, cv2.TM_CCOEFF_NORMED)
    _, max_amt, _, max_loc = cv2.minMaxLoc(res)
    if max_amt > 0.999:
        return max_loc[::-1]
    return None


def get_target_coords_naive(target, array):
    for ix, iy in np.ndindex(array.shape[0] - target.shape[0] + 1, array.shape[1] - target.shape[1] + 1):
        curr_subarray = array[ix:ix + target.shape[0], iy:iy + target.shape[1]]
        if np.array_equal(target, curr_subarray):
            return ix, iy
    return None


def get_following_characters(target, array, coords=(0, 0)):
    return array[
           coords[0]:coords[0] + target.shape[0],
           coords[1] + target.shape[1]:coords[1] + target.shape[1] + 5
        ], array[
           coords[0]:coords[0] + target.shape[0],
           coords[1] + target.shape[1] + 6:coords[1] + target.shape[1] + 11
        ]


class GameState:
    def __init__(self):
        self.templates = {}
        for filename in os.listdir("templates"):
            match filename:
                case "colon.bmp":
                    key = ":"
                case "space.bmp":
                    key = " "
                case _:
                    key = filename.split(".")[0]
            self.templates[key] = cv2.imread(f'templates/{filename}', cv2.IMREAD_GRAYSCALE) // NUM_COLORS

        self.lookup_dict = {}
        for character, char_array in self.templates.items():
            array_hash = hash("".join(char_array.flatten().astype(str)))
            self.lookup_dict[array_hash] = int(character) if character.isdecimal() else character

        self.sct = mss()

        screen = cv2.cvtColor(np.array(self.sct.grab(self.sct.monitors[1])), cv2.COLOR_BGRA2GRAY)
        self.lv_target = self.templates["LV"]
        self.top_corner = get_target_coords(self.lv_target, screen)

        pointer = [self.top_corner[0] + 100, self.top_corner[1]]
        while screen[pointer[0], pointer[1]] > 10 or screen[pointer[0] + 1, pointer[1]] < 250:
            pointer[0] += 1
        while screen[pointer[0], pointer[1]] > 10 or screen[pointer[0], pointer[1] + 1] < 250:
            pointer[1] += 1
        self.bottom_corner = (pointer[0], pointer[1])

        self.screen = np.array([])
        self.level = 0
        self.update_game_state()
        self.monster_counts = self.get_monster_counts()

    def update_game_state(self):
        img = np.array(self.sct.grab(self.sct.monitors[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        img = img[self.top_corner[0]:self.bottom_corner[0] + 1, self.top_corner[1]:self.bottom_corner[1] + 1]

        img = img // NUM_COLORS
        lv_array, _ = get_following_characters(self.lv_target[::4, ::4], img[::4, ::4])
        img = img[::2, ::2]
        self.screen = img
        self.level = self.get_value(lv_array)

    def get_monster_counts(self):
        count_array = []
        for i in range(1, 10):
            count_target = self.str_to_array(f"LV{i}:x")
            count_coords = get_target_coords(count_target, self.screen)
            if not count_coords:
                break
            num_array, next_num_array = get_following_characters(count_target, self.screen, count_coords)
            val1 = self.get_value(num_array)
            val2 = self.get_value(next_num_array)
            if val2 != " ":
                val1 = 10 * val1 + val2
            count_array.append(val1)
        return count_array

    def str_to_array(self, to_convert):
        vert_break = np.zeros((7, 1), dtype=np.uint8)
        char_arrays = []
        for character in to_convert:
            char_arrays.append(self.templates[character])
            char_arrays.append(vert_break)

        return np.hstack(char_arrays)

    def get_value(self, array):
        key = hash("".join(array.flatten().astype(str)))
        return self.lookup_dict[key]


def main():
    game = GameState()
    print(game.level, game.monster_counts)
    print(game.screen)


if __name__ == '__main__':
    main()
