import numpy as np
from scipy.signal import correlate2d

from TASweeper.game_state import GameState
from TASweeper.utils import clickable_set


# Click any unrevealed tiles next to neighbor counts less than or equal to current level
def get_low_neighbor_count(game: GameState) -> clickable_set:
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    # unrevealed_neighbors is 8 if all neighbors are unrevealed (boring) or 0 if all are revealed (boring)
    unrevealed_neighbors = correlate2d(
        game.unrevealed, kernel, boundary="symm", mode="same"
    )

    # Account for values which are high due to known monsters
    known_value_modifier = correlate2d(game.grid_values, kernel, mode="same")
    modified_neighbor_count = game.neighbor_count - known_value_modifier

    # must have a neighbor with a number
    number_boundary = np.all((unrevealed_neighbors > 0, ~game.unrevealed), axis=0)
    contacting_nums_under_level = np.all(
        (number_boundary, game.neighbor_known, modified_neighbor_count <= game.level),
        axis=0,
    )
    neighbor_under_level = (
        correlate2d(contacting_nums_under_level, kernel, mode="same") > 0
    )
    safe_to_click = np.all(
        (neighbor_under_level, game.unrevealed, game.grid_values <= game.level), axis=0
    )
    x, y = np.where(safe_to_click)
    return set(zip(x, y))
