import numpy as np
from scipy.signal import correlate2d

from TASweeper.game_state import GameState
from TASweeper.utils import clickable_set


# Click any unrevealed tiles next to neighbor counts less than or equal to current level
def get_low_neighbor_count(game: GameState) -> clickable_set:
    nums_under_level = np.all(
        (game.neighbor_known, game.modified_neighbor_count <= game.level),
        axis=0,
    )
    # must have a neighbor under level
    contacting_num_under_level = (
        correlate2d(
            nums_under_level, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), mode="same"
        )
        > 0
    )
    safe_to_click = np.all(
        (contacting_num_under_level, game.unrevealed, game.grid_values <= game.level),
        axis=0,
    )
    x, y = np.where(safe_to_click)
    return set(zip(x, y))
