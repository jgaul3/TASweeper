import numpy as np

from TASweeper.game_state import GameState
from TASweeper.utils import clickable_set


def get_known_under_level(game: GameState) -> clickable_set:
    # Get any previously-identified spaces below level
    x, y = np.where(
        np.all(
            (game.unrevealed, game.grid_values <= game.level, game.grid_value_known),
            axis=0,
        )
    )
    return set(zip(x, y))
