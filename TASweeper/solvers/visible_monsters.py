import numpy as np
from scipy.signal import correlate2d

from TASweeper.game_state import GameState
from TASweeper.utils import clickable_set


# Click all monsters which are next to unrevealed squares.
def get_visible_monsters(game: GameState) -> clickable_set:
    # unrevealed_neighbors is > 0 if tile is next to unknown grid value
    unrevealed_neighbors = correlate2d(
        ~game.grid_value_known, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), mode="same"
    )

    to_click = np.all(
        (
            # Only consider revealed tiles
            ~game.unrevealed,
            # Ignore 0 tiles
            game.grid_values > 0,
            # Only consider tiles where neighbor count is unknown
            ~game.neighbor_known,
            # Must have unknown neighbor
            unrevealed_neighbors > 0,
        ),
        axis=0,
    )
    x, y = np.where(to_click)
    return set(zip(x, y))
