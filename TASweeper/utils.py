import os
from typing import Set, Tuple

import cv2
import numpy as np

from numpy_dict import NumpyDict


# cv2.imwrite(
#     "asdf.bmp",
#     np.array(cv2.cvtColor(
#     np.array(self.sct.grab(self.sct.monitors[1])), cv2.COLOR_BGRA2GRAY
# ) > 60, dtype=np.uint8) * 255
# )

HIGH_THRESHOLD = 60
THRESHOLD = 20

clickable_set = Set[Tuple[int, int]]


def populate_templates():
    template_dict = {}
    for filename in os.listdir("TASweeper/templates"):
        if filename.startswith("."):
            continue
        key = filename.split(".")[0]
        template_dict[key] = (
            cv2.imread(f"TASweeper/templates/{filename}", cv2.IMREAD_GRAYSCALE)
            > HIGH_THRESHOLD
        )
    return template_dict


def populate_lookup_dict():
    lookup_dict = NumpyDict()

    # Unchecked (all bright) and empty (all dark) grids should read as 0
    char_array = np.ones((7, 14), dtype=bool)
    lookup_dict[char_array] = (0, 0)
    char_array = np.zeros((7, 14), dtype=bool)
    lookup_dict[char_array] = (0, 0)

    # Number in center of "##" width
    for i in range(10):
        char_array = np.zeros((7, 14), dtype=bool)
        char_array[:, 4:10] = templates[str(i)]
        lookup_dict[char_array] = (i, 0)

    # "##" combos
    for i in range(73):
        digit_string = f"{i:02d}"
        char_array = np.hstack(
            (
                templates[str(digit_string[0])],
                np.zeros((7, 2), dtype=bool),
                templates[str(digit_string[1])],
            )
        )
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
    if max_amt < 0.999:
        raise Exception("Image not found")
    return max_loc[::-1]


templates = populate_templates()
lookup_dict = populate_lookup_dict()
