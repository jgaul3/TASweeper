import numpy as np
from scipy.signal import correlate2d

from TASweeper.game_state import GameState
from TASweeper.utils import clickable_set


# Click all monsters which are next to unrevealed squares.
def get_visible_monsters(game: GameState) -> clickable_set:
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    # unrevealed_neighbors is 8 if all neighbors are unrevealed (boring) or 0 if all are revealed (boring)
    unrevealed_neighbors = correlate2d(
        game.unrevealed, kernel, boundary="symm", mode="same"
    )

    # Monsters must have unrevealed neighbors
    x, y = np.where(
        np.all(
            (
                ~game.unrevealed,
                game.grid_values > 0,
                ~game.neighbor_known,
                unrevealed_neighbors > 0,
            ),
            axis=0,
        )
    )
    return set(zip(x, y))
