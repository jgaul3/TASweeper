import numpy as np
from scipy.signal import convolve2d

from ..utils import clickable_set


# Click any unrevealed tiles next to neighbor counts less than or equal to current level
def get_unrevealed_under_level(self) -> clickable_set:
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
