from random import randrange

import numpy as np

from TASweeper.game_state import GameState
from TASweeper.utils import clickable_set


def get_random_unrevealed(game: GameState) -> clickable_set:
    x, y = np.where(np.all((game.unrevealed, game.grid_values <= game.level), axis=0))
    point = randrange(len(x))
    return {(x[point], y[point])}
