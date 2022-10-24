import time

from TASweeper.solvers.low_neighbor_count import get_unrevealed_under_level
from TASweeper.solvers.random_unrevealed import get_random_unrevealed
from game_state import GameState


def main():
    game = GameState()
    while True:
        try:
            game.click_grid(-1, 0, 2)
            to_click = get_random_unrevealed(game)
            while True:
                # game.click_grid(-1, 0)  # Focus game if debugging
                while to_click:
                    game.click_grid(*to_click.pop())

                # Start simple, get difficult, end with a random guess (weighted?)
                to_click = (
                    game.update_game_state()
                    or get_unrevealed_under_level(game)
                    or get_random_unrevealed(game)
                )
                print(to_click)

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
