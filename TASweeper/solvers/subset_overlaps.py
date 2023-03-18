import numpy as np
from scipy.signal import correlate2d

from TASweeper.solvers.derive_lone_values import derive_lone_values
from TASweeper.game_state import GameState
from TASweeper.utils import clickable_set


def get_subset_overlaps(game: GameState) -> clickable_set:
    set_array = np.array(
        [set() for _ in range(len(game.unrevealed.flatten()))]
    ).reshape(game.unrevealed.shape)

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    # unrevealed_neighbors is 8 if all neighbors are unrevealed (boring) or 0 if all are revealed (boring)
    unrevealed_neighbors = correlate2d(
        game.unrevealed, kernel, boundary="symm", mode="same"
    )

    edges = np.all(
        (
            unrevealed_neighbors > 0,
            unrevealed_neighbors < 8,
            game.unrevealed,
            ~game.grid_value_known,
        ),
        axis=0,
    )
    edges = np.pad(edges, 1)

    for (x, y), _ in np.ndenumerate(set_array):
        xs, ys = np.where(edges[x : x + 3, y : y + 3])
        set_array[x, y] = set(zip(xs + x - 1, ys + y - 1))

    known_value_modifier = correlate2d(game.grid_values, kernel, mode="same")
    modified_neighbor_count = game.neighbor_count - known_value_modifier

    neighbor_squares = np.all(
        (
            unrevealed_neighbors > 0,
            unrevealed_neighbors < 8,
            game.neighbor_known,
            set_array != set(),
        ),
        axis=0,
    )
    to_click = set()
    updated_values = False
    for i1 in zip(*np.where(neighbor_squares)):
        relevant_set_mask = np.zeros(
            (neighbor_squares.shape[0] + 4, neighbor_squares.shape[1] + 4), dtype=bool
        )
        relevant_set_mask[i1[0] : i1[0] + 5, i1[1] : i1[1] + 5] = True
        relevant_set_mask = relevant_set_mask[2:-2, 2:-2]
        relevant_set_mask = np.all((neighbor_squares, relevant_set_mask), axis=0)
        for i2 in zip(*np.where(relevant_set_mask)):
            if set_array[i1] > set_array[i2]:
                # check other elements in i1
                diff = set_array[i1] - set_array[i2]
                obligate_neighbor_count = (
                    modified_neighbor_count[i1] - modified_neighbor_count[i2]
                )
                if obligate_neighbor_count < 0 or obligate_neighbor_count > 100:
                    return set()
                if obligate_neighbor_count <= game.level:
                    to_click = to_click.union(diff)
                elif len(diff) == 1:
                    updated_values = True
                    outsider = diff.pop()
                    game.grid_values[outsider] = obligate_neighbor_count
                    game.grid_value_known[outsider] = True

    if updated_values:
        derive_lone_values(game)

    return to_click
