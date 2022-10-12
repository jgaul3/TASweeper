import time
import os
import cv2
import numpy as np
import pyautogui
from mss import mss


THRESHOLD = 10


def get_target_coords(target, array):
    res = cv2.matchTemplate(
        array.astype(np.uint8), target.astype(np.uint8), cv2.TM_CCOEFF_NORMED
    )
    _, max_amt, _, max_loc = cv2.minMaxLoc(res)
    return max_loc[::-1] if max_amt > 0.999 else None


# Dict which accepts np arrays as keys
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

        self.top_corner = self.bottom_corner = (0, 0)
        self.screen = self.board = self.level = self.hit_points = None
        self.update_game_state(initialize=True)
        self.grid_top_left, self.grid_bottom_right, self.grid_width = self.map_grid()
        height = (self.grid_bottom_right[0] - self.grid_top_left[0]) // self.grid_width
        width = (self.grid_bottom_right[1] - self.grid_top_left[1]) // self.grid_width
        self.grid_values = np.zeros((height, width), dtype=np.uint8)
        self.neighboring_counts = np.zeros((height, width), dtype=np.uint8)
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
        for i in [True, False]:
            char_array = i and np.ones((7, 12), dtype=bool)
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

        if initialize:
            return

        new_board = self.screen[
            self.grid_top_left[0] : self.grid_bottom_right[0],
            self.grid_top_left[1] : self.grid_bottom_right[1],
        ]
        for (x, y), value in np.ndenumerate(self.neighboring_counts):
            number_array = self.board[
                16 * x + 4 : 16 * x + 11, 16 * y + 2 : 16 * y + 14
            ]
            neighbors, own_count = self.lookup_dict[number_array]
            self.neighboring_counts[x, y] = neighbors
            self.is_revealed[x, y] = 0 in number_array

    def get_monster_counts(self):
        count_array = [len(self.is_revealed.flatten())]
        for i in range(1, 10):
            count_target = self.str_to_array(f"LV{i}:x")
            count_coords = get_target_coords(count_target, self.screen)
            if count_coords:
                monster_count = self.get_following_value(5, self.screen, count_coords)
                count_array.append(monster_count)
                count_array[0] -= monster_count
            else:
                count_array.append(0)

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
        while self.screen[pointer[0], pointer[1]]:
            pointer[0] += 1
        width = pointer[0] - top_left[0] + 1
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
        bottom_right = [pointer[0] + 2, pointer[1] + 2]
        return top_left, bottom_right, width

    def click_grid(self, x, y):
        pyautogui.click(
            clicks=2,
            x=self.top_corner[1] // 2
            + self.grid_top_left[1]
            + self.grid_width * y
            + self.grid_width // 2,
            y=self.top_corner[0] // 2
            + self.grid_top_left[0]
            + self.grid_width * x
            + self.grid_width // 2,
        )

    def solve(self):
        action_queue = []
        while True:
            if not action_queue:
                pass  # Click a random (weighted?) unrevealed square
            else:
                # Execute actions in queue
                pass
            self.update_game_state()
            # Update state
            # If HP is zero -> exit
            # Reveal monster values
            # Update state
            # Fill action queue
            #


def main():
    game = GameState()
    game.solve()


if __name__ == "__main__":
    main()
