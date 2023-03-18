import pickle
import sys
from copy import copy

from game_state import GameState, Win, NoHitPoints
from solvers.derive_lone_values import derive_lone_values
from solvers.known_under_level import get_known_under_level
from solvers.low_neighbor_count import get_low_neighbor_count
from solvers.subset_overlaps import get_subset_overlaps
from solvers.random_unrevealed import get_random_unrevealed
from solvers.visible_monsters import get_visible_monsters


def solve(game: GameState):
    to_click = get_random_unrevealed(game)
    while True:
        to_click = sorted(list(to_click))
        while to_click:
            game.click_grid(*to_click.pop(0))

        game.update_game_state()

        derive_lone_values(game)

        to_click = (
            # Try to reveal more of the grid
            get_subset_overlaps(game)
            | get_low_neighbor_count(game)
            | get_known_under_level(game)
            # Not enough info, try clicking revealed monsters
            or get_visible_monsters(game)
            # Still no info, just guess
            or get_random_unrevealed(game)
        )


def debug_solve(game: GameState):
    to_click_type = "random"
    max_hp = game.hit_points
    trace = []
    to_click = get_random_unrevealed(game)
    try:
        while True:
            trace.append(dict())
            to_click = sorted(list(to_click))

            trace[-1]["initial_state"] = game.copy_state()
            trace[-1]["to_click"] = copy(to_click)
            trace[-1]["to_click_type"] = to_click_type

            # debug_logging(game, to_click, len(trace))
            while to_click:
                game.click_grid(*to_click.pop(0))
                # debug_clicking(game, max_hp)

            game.update_game_state()
            trace[-1]["updated_game"] = game.copy_state()

            derive_lone_values(game)
            trace[-1]["derived_game"] = game.copy_state()

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

            if game.hit_points < max_hp:
                print("huh")
    except Win:
        raise
    except Exception:
        with open("test_game.pickle", "wb") as file:
            pickle.dump(trace, file)
        raise


def main():
    tries = 20
    while True:
        try:
            game = GameState()
            engine = solve if sys.gettrace() is None else debug_solve
            engine(game)
        except NoHitPoints:
            print(f"rip lol, {tries} tries left")
            tries -= 1
            if tries == 0:
                break
        except Exception as e:
            print(e)
            if input("Continue? y/n") != "y":
                break


if __name__ == "__main__":
    main()
