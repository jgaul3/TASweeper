from types import SimpleNamespace

import numpy as np

import pickle

from TASweeper.pain import pain

if __name__ == "__main__":
    # test_game = SimpleNamespace(
    #     unrevealed=np.array([
    #         [0, 1, 0, 1, 1],
    #         [0, 1, 0, 1, 1],
    #         [0, 1, 0, 1, 1],
    #     ], dtype=bool),
    #     grid_values=np.array([
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0],
    #     ], dtype=np.uint8),
    #     grid_value_known=np.array([
    #         [1, 0, 1, 0, 0],
    #         [1, 0, 1, 0, 0],
    #         [1, 0, 1, 0, 0],
    #     ], dtype=bool),
    #     neighbor_count=np.array([
    #         [1, 0, 1, 0, 0],
    #         [1, 0, 1, 0, 0],
    #         [1, 0, 1, 0, 0],
    #     ], dtype=np.uint8),
    #     neighbor_known=np.array([
    #         [1, 0, 1, 0, 0],
    #         [1, 0, 1, 0, 0],
    #         [1, 0, 1, 0, 0],
    #     ], dtype=bool),
    #     level=1,
    # )
    #
    # # noinspection PyTypeChecker
    # to_click = pain(test_game)
    # assert to_click == {
    #     (0, 1), (2, 1), (0, 3), (1, 3), (2, 3)
    # }
    #
    # test_game = SimpleNamespace(
    #     unrevealed=np.array([
    #         [0, 1, 0, 1, 1],
    #         [0, 1, 0, 1, 1],
    #     ], dtype=bool),
    #     grid_values=np.array([
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0],
    #     ], dtype=np.uint8),
    #     grid_value_known=np.array([
    #         [1, 0, 1, 0, 0],
    #         [1, 0, 1, 0, 0],
    #     ], dtype=bool),
    #     neighbor_count=np.array([
    #         [1, 0, 1, 0, 0],
    #         [1, 0, 1, 0, 0],
    #     ], dtype=np.uint8),
    #     neighbor_known=np.array([
    #         [1, 0, 1, 0, 0],
    #         [1, 0, 1, 0, 0],
    #     ], dtype=bool),
    #     level=1,
    # )
    #
    # # noinspection PyTypeChecker
    # to_click = pain(test_game)
    # assert to_click == {
    #     (0, 3), (1, 3)
    # }

    # test_game = SimpleNamespace(
    #     unrevealed=np.array([
    #         [0, 0, 0, 0],
    #         [1, 1, 1, 1],
    #     ], dtype=bool),
    #     grid_values=np.array([
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #     ], dtype=np.uint8),
    #     grid_value_known=np.array([
    #         [1, 1, 1, 1],
    #         [0, 0, 0, 0],
    #     ], dtype=bool),
    #     neighbor_count=np.array([
    #         [3, 5, 2, 2],
    #         [0, 0, 0, 0],
    #     ], dtype=np.uint8),
    #     neighbor_known=np.array([
    #         [1, 1, 1, 1],
    #         [0, 0, 0, 0],
    #     ], dtype=bool),
    #     level=1,
    # )
    #
    # # noinspection PyTypeChecker
    # to_click = pain(test_game)
    # assert to_click == {
    #     (0, 3), (1, 3)
    # }

    # test_game = SimpleNamespace(
    #     unrevealed=np.array([
    #         [1, 1, 0, 0, 0, 0],
    #         [1, 1, 0, 0, 0, 0],
    #         [1, 1, 0, 0, 0, 0],
    #         [1, 1, 1, 0, 0, 0],
    #         [1, 1, 1, 1, 1, 1],
    #     ], dtype=bool),
    #     grid_values=np.array([
    #         [0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0],
    #         [0, 0, 4, 0, 0, 0],
    #         [0, 0, 0, 0, 2, 1],
    #     ], dtype=np.uint8),
    #     grid_value_known=np.array([
    #         [0, 0, 1, 1, 1, 1],
    #         [0, 0, 1, 1, 1, 1],
    #         [0, 0, 1, 1, 1, 1],
    #         [0, 0, 1, 1, 1, 1],
    #         [0, 0, 1, 1, 1, 1],
    #     ], dtype=bool),
    #     neighbor_count=np.array([
    #         [0, 0, 2, 0, 0, 0],
    #         [0, 0, 2, 0, 0, 0],
    #         [0, 0, 4, 4, 0, 0],
    #         [0, 0, 0, 6, 3, 3],
    #         [0, 0, 0, 0, 0, 0],
    #     ], dtype=np.uint8),
    #     neighbor_known=np.array([
    #         [0, 0, 1, 1, 1, 1],
    #         [0, 0, 1, 1, 1, 1],
    #         [0, 0, 1, 1, 1, 1],
    #         [0, 0, 0, 1, 1, 1],
    #         [0, 0, 0, 0, 0, 0],
    #     ], dtype=bool),
    #     level=1,
    # )
    with open("test_game.pickle", "rb") as file:
        test_game = pickle.load(file)
    # noinspection PyTypeChecker
    to_click = pain(test_game)
    print(to_click)
