from random import randrange

import numpy as np

from ..game_state import GameState
from ..utils import clickable_set


def get_random_unrevealed(game: GameState) -> clickable_set:
    x, y = np.where(
        game.unrevealed
    )  # TODO: make sure you don't pick a known high value
    point = randrange(len(x))
    return {(x[point], y[point])}
