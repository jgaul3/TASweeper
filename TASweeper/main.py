import sys
import time

from solvers.known_under_level import get_known_under_level
from solvers.low_neighbor_count import get_low_neighbor_count
from solvers.random_unrevealed import get_random_unrevealed
from solvers.visible_monsters import get_visible_monsters
from game_state import GameState


def main():
    game = GameState()
    while True:
        try:
            game.click_grid(-1, 0, 2)
            to_click = get_random_unrevealed(game)
            while True:
                to_click = sorted(list(to_click))
                if sys.gettrace() is not None:
                    print(to_click)
                    game.click_grid(-1, 0, 2)  # Focus game if debugging
                while to_click:
                    game.click_grid(*to_click.pop(0))

                game.update_game_state()

                game.derive_lone_values()

                # Start simple, get difficult, end with a random guess (weighted?)
                to_click = (
                    (get_low_neighbor_count(game) | get_known_under_level(game))
                    or get_visible_monsters(
                        game
                    )  # Not enough info, try revealing monsters
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


if __name__ == "__main__":
    main()
