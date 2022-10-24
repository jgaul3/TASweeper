import os
from typing import Set, Tuple

import cv2
import numpy as np
import pyautogui

from numpy_dict import NumpyDict


# pyautogui.DARWIN_CATCH_UP_TIME = 0.005
pyautogui.PAUSE = 0
THRESHOLD = 10

clickable_set = Set[Tuple[int, int]]


def populate_templates():
    template_dict = {}
    for filename in os.listdir("TASweeper/templates"):
        match filename:
            case "colon.bmp":
                key = ":"
            case "space.bmp":
                key = " "
            case _:
                key = filename.split(".")[0]
        template_dict[key] = (
            cv2.imread(f"TASweeper/templates/{filename}", cv2.IMREAD_GRAYSCALE)
            > THRESHOLD
        )
    return template_dict


def str_to_array(to_convert):
    vert_break = np.zeros((7, 1), dtype=bool)
    char_arrays = []
    for character in to_convert:
        char_arrays.append(templates[character])
        char_arrays.append(vert_break)

    return np.hstack(char_arrays)


def populate_lookup_dict():
    lookup_dict = NumpyDict()

    # Unchecked (all bright) and empty (all dark) grids should read as 0
    char_array = np.ones((7, 12), dtype=bool)
    lookup_dict[char_array] = (0, 0)
    char_array = np.zeros((7, 12), dtype=bool)
    lookup_dict[char_array] = (0, 0)

    # "# " should return the number
    for i in range(10):
        char_array = str_to_array(f"{i} ")
        lookup_dict[char_array] = (i, 0)

    # Number combos incl. in center of patch
    for i in range(73):
        char_array = str_to_array(f"{i}")
        if i < 10:
            base = np.zeros((7, 12), dtype=bool)
            base[:, 3:9] = char_array
            char_array = base
        lookup_dict[char_array] = (i, 0)

    # Monsters
    for i in range(1, 10):
        char_array = templates[f"mon_{i}"]
        lookup_dict[char_array] = (0, i)

    return lookup_dict


def get_target_coords(target, array):
    res = cv2.matchTemplate(
        array.astype(np.uint8), target.astype(np.uint8), cv2.TM_CCOEFF_NORMED
    )
    _, max_amt, _, max_loc = cv2.minMaxLoc(res)
    return max_loc[::-1] if max_amt > 0.999 else None


templates = populate_templates()
lookup_dict = populate_lookup_dict()
