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
        self.populate_templates()

        self.lookup_dict = {}
        self.grid_lookup_dict = {}
        self.populate_lookup_dicts()

        self.sct = mss()

        self.lv_target = self.templates["LV"]
        self.top_corner, self.bottom_corner = self.find_game_screen_coordinates(
            cv2.cvtColor(np.array(self.sct.grab(self.sct.monitors[1])), cv2.COLOR_BGRA2GRAY) // NUM_COLORS
        )

        self.screen = None
        self.level = 0
        self.update_game_state(initialize=True)
        self.grid_top_left, self.grid_bottom_right, self.grid_width = self.map_grid()
        height = (self.grid_bottom_right[0] - self.grid_top_left[0]) // self.grid_width
        width = (self.grid_bottom_right[1] - self.grid_top_left[1]) // self.grid_width
        self.neighboring_counts = np.zeros((height, width), dtype=int)
        self.is_revealed = np.zeros((height, width), dtype=bool)
        self.monster_counts = self.get_monster_counts(width * height)

    def populate_templates(self):
        for filename in os.listdir("templates"):
            match filename:
                case "colon.bmp":
                    key = ":"
                case "space.bmp":
                    key = " "
                case _:
                    key = filename.split(".")[0]
            self.templates[key] = cv2.imread(f'templates/{filename}', cv2.IMREAD_GRAYSCALE) // NUM_COLORS

    def populate_lookup_dicts(self):
        for character, char_array in self.templates.items():
            array_hash = hash("".join(char_array.flatten().astype(str)))
            self.lookup_dict[array_hash] = int(character) if character.isdecimal() else character

        for i in range(255 // NUM_COLORS):
            base = np.ones((7, 12), dtype=int)
            base *= i
            array_hash = hash("".join(base.flatten().astype(str)))
            self.grid_lookup_dict[array_hash] = 0

        for i in range(73):
            if i < 10:
                base = np.zeros((7, 12), dtype=int)
                number = self.str_to_array(f"{i}")
                base[:, 3:9] = number
            else:
                base = self.str_to_array(f"{i}")
            array_hash = hash("".join(base.flatten().astype(str)))
            self.grid_lookup_dict[array_hash] = i

    def find_game_screen_coordinates(self, full_screen):
        top_corner = get_target_coords(self.lv_target, full_screen)

        pointer = [top_corner[0] + 100, top_corner[1]]
        while full_screen[pointer[0], pointer[1]] > 0 or full_screen[pointer[0] + 1, pointer[1]] < (NUM_COLORS - 1):
            pointer[0] += 1
        while full_screen[pointer[0], pointer[1]] > 0 or full_screen[pointer[0], pointer[1] + 1] < (NUM_COLORS - 1):
            pointer[1] += 1
        bottom_corner = (pointer[0], pointer[1])
        return top_corner, bottom_corner

    def update_game_state(self, initialize=False):
        img = cv2.cvtColor(np.array(self.sct.grab(self.sct.monitors[1])), cv2.COLOR_BGRA2GRAY)
        img = img[self.top_corner[0]:self.bottom_corner[0] + 1, self.top_corner[1]:self.bottom_corner[1] + 1]

        img = img // NUM_COLORS
        lv_array, _ = get_following_characters(self.lv_target[::4, ::4], img[::4, ::4])
        img = img[::2, ::2]
        self.screen = img
        self.level = self.get_single_value(lv_array)

        if initialize:
            return

        board = self.screen[self.grid_top_left[0]:self.grid_bottom_right[0], self.grid_top_left[1]:self.grid_bottom_right[1]]
        for (x, y), value in np.ndenumerate(self.neighboring_counts):
            number_array = board[16 * x + 4:16 * x + 11, 16 * y + 2:16 * y + 14]
            self.neighboring_counts[x, y] = self.get_grid_value(number_array)
            self.is_revealed[x, y] = number_array[-1, -1] > 0

    def get_monster_counts(self, blanks):
        count_array = [blanks]
        for i in range(1, 10):
            count_target = self.str_to_array(f"LV{i}:x")
            count_coords = get_target_coords(count_target, self.screen)
            if not count_coords:
                break
            num_array, next_num_array = get_following_characters(count_target, self.screen, count_coords)
            val1 = self.get_single_value(num_array)
            val2 = self.get_single_value(next_num_array)
            if val2 != " ":
                val1 = 10 * val1 + val2
            count_array.append(val1)
            count_array[0] -= val1

        return count_array

    def str_to_array(self, to_convert):
        vert_break = np.zeros((7, 1), dtype=np.uint8)
        char_arrays = []
        for character in to_convert:
            char_arrays.append(self.templates[character])
            char_arrays.append(vert_break)

        return np.hstack(char_arrays)

    def get_single_value(self, array):
        key = hash("".join(array.flatten().astype(str)))
        return self.lookup_dict.get(key, 0)

    def get_grid_value(self, array):
        array[array > 0] = 255 // NUM_COLORS
        key = hash("".join(array.flatten().astype(str)))
        return self.grid_lookup_dict.get(key, 0)

    def map_grid(self):
        pointer = [0, 0]
        while self.screen[pointer[0], pointer[1]] > 0:
            pointer[0] += 1
        pointer[0] += 1
        while self.screen[pointer[0], pointer[1]] == 0:
            pointer[1] += 1
        top_left = pointer.copy()
        while self.screen[pointer[0], pointer[1]] > 0:
            pointer[0] += 1
        width = pointer[0] - top_left[0] + 1
        while self.screen[pointer[0] + 1, pointer[1]] > 0 or self.screen[pointer[0] + 2, pointer[1]] > 0:
            pointer[0] += 1
        while self.screen[pointer[0], pointer[1] + 1] > 0 or self.screen[pointer[0], pointer[1] + 2] > 0:
            pointer[1] += 1
        bottom_right = [pointer[0] + 2, pointer[1] + 2]
        return top_left, bottom_right, width

    def click_grid(self, x, y, home=True):
        pyautogui.click(
            clicks=2,
            x=self.top_corner[1] // 2 + self.grid_top_left[1] + self.grid_width * y + self.grid_width // 2,
            y=self.top_corner[0] // 2 + self.grid_top_left[0] + self.grid_width * x + self.grid_width // 2
        )
        if home:
            pyautogui.moveTo(
                self.bottom_corner[1] // 2,
                self.bottom_corner[0] // 2
            )

    def solve(self):
        action_queue = []
        while True:
            if not action_queue:
                pass  # Click a random (weighted?) unrevealed square
            else:
                # Execute actions in queue
                # Update state
                # Fill action queue
                pass


def main():
    game = GameState()
    game.solve()


if __name__ == '__main__':
    main()
