import numpy as np
from scipy.signal import convolve2d


from TASweeper.game_state import GameState


# Fast way to identify squares where neighbor count
# can only come from a single neighbor
def derive_lone_values(game: GameState):
    found_value = True
    while found_value:
        found_value = False
        power_kernel = np.array([[1, 2, 4], [8, 0, 16], [32, 64, 128]])

        power_mapping = {
            1: (1, 1),
            2: (1, 0),
            4: (1, -1),
            8: (0, 1),
            0: (0, 0),
            16: (0, -1),
            32: (-1, 1),
            64: (-1, 0),
            128: (-1, -1),
        }

        power_array = convolve2d(~game.grid_value_known, power_kernel, mode="same")
        has_one_unknown_neighbor = np.all(
            (
                # power_array is a power of 2
                np.bitwise_and(power_array, power_array - 1) == 0,
                # power_array is adjacent to an unknown value
                power_array != 0,
                # neighbor value is known
                game.neighbor_known,
            ),
            axis=0,
        )

        # Set all others to zero, only dealing with powers of 2
        power_array[~has_one_unknown_neighbor] = 0

        indices_array = np.indices(power_array.shape).transpose(1, 2, 0)
        for key in power_mapping:
            indices_array[power_array == key] += power_mapping[key]

        known_value_modifier = convolve2d(
            game.grid_values, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), mode="same"
        )
        modified_neighbor_count = game.neighbor_count - known_value_modifier

        indices_and_count = np.dstack((indices_array, modified_neighbor_count))
        loner_squares = indices_and_count[np.where(power_array != 0)]
        coord_value_set = set(map(tuple, loner_squares))

        if any(value[2] < 0 for value in coord_value_set):
            test = 1

        if coord_value_set:
            found_value = True
            for x, y, value in coord_value_set:
                game.grid_values[x, y] = value
                game.grid_value_known[x, y] = True
