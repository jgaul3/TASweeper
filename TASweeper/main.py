import sys
import time

import pyautogui

from debugging import debug_logging, debug_clicking
from derive_lone_values import derive_lone_values
from game_state import GameState
from solvers.known_under_level import get_known_under_level
from solvers.low_neighbor_count import get_low_neighbor_count
from solvers.random_unrevealed import get_random_unrevealed
from solvers.visible_monsters import get_visible_monsters


def main():
    game = GameState()
    while True:
        try:
            game.click_grid(-1, 0, 2)
            to_click = get_random_unrevealed(game)
            while True:
                to_click = sorted(list(to_click))
                while to_click:
                    game.click_grid(*to_click.pop(0))

                game.update_game_state()

                derive_lone_values(game)

                clickable_squares = get_low_neighbor_count(
                    game
                ) | get_known_under_level(game)
                to_click = (
                    # Try to reveal more of the grid
                    clickable_squares
                    # Not enough info, try clicking revealed monsters
                    or get_visible_monsters(game)
                    # Still no info, just guess
                    or get_random_unrevealed(game)
                )

        except Exception as e:
            print(e)
            if input("Continue? y/n") == "y":
                game.click_grid(-1, 0, 2)
                time.sleep(0.1)
                game.__init__()
            else:
                break


def debug_main():
    pyautogui.PAUSE = 0.1
    game = GameState()
    while True:
        try:
            to_click_type = "random"
            max_hp = game.hit_points
            trace = []
            game.click_grid(-1, 0, 2)
            to_click = {}
            while True:
                trace.append(dict())
                to_click = sorted(list(to_click))

                trace[-1]["initial_state"] = game.copy_state()
                trace[-1]["to_click"] = to_click
                trace[-1]["to_click_type"] = to_click_type

                debug_logging(game, to_click, len(trace))
                while to_click:
                    game.click_grid(*to_click.pop(0))
                    debug_clicking(game, max_hp)

                game.update_game_state()
                trace[-1]["updated_game"] = game.copy_state()

                derive_lone_values(game)
                trace[-1]["derived_game"] = game.copy_state()

                # Start simple, get difficult, end with a random guess (weighted?)
                if clickable_squares := (
                    get_low_neighbor_count(game) | get_known_under_level(game)
                ):
                    to_click_type = "clickables"
                    to_click = clickable_squares
                elif visibles := get_visible_monsters(game):
                    to_click_type = "vis"
                    to_click = visibles
                else:
                    to_click_type = "random"
                    to_click = get_random_unrevealed(game)

        except Exception as e:
            print(e)
            if input("Continue? y/n") == "y":
                game.click_grid(-1, 0, 2)
                time.sleep(0.1)
                game.__init__()
            else:
                break


if __name__ == "__main__":
    if sys.gettrace() is None:
        main()
    else:
        debug_main()
