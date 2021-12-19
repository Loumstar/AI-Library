import json

def sum_times_and_states(game_version):
    """
    Function to sum the times and states of the players for a game.
    ---
    Useful for putting into a table for the report.
    """
    game_version["sums"] = {
        "max": {
            "time": sum([a["time"] for a in game_version["max"]]),
            "states": sum([a["visited_states"] for a in game_version["max"]])
        },
        "min": {
            "time": sum([a["time"] for a in game_version["min"]]),
            "states": sum([a["visited_states"] for a in game_version["min"]])
        },
        "game_time": sum([a["time"] for a in game_version["max"]]) \
                    + sum([a["time"] for a in game_version["min"]]),
        "game_states": sum([a["visited_states"] for a in game_version["max"]]) \
                    + sum([a["visited_states"] for a in game_version["min"]])
    }

    return game_version

def print_numbers_to_latex_table(game_version):
    """
    Function to print the latex syntax for a table without having to
    actually write it out by hand.
    """
    parameters = game_version["parameters"]

    m = parameters.get("columns")
    n = parameters.get("rows")
    k = parameters.get("win_length")

    time = game_version["sums"].get("game_time")
    states = game_version["sums"].get("game_states")

    print(f"{m} & {n} & {k} & {1000 * time:.3f} & {states} \\\\")

if __name__ == "__main__":
    with open('results/minimax_time_results.json', 'r') as fp:
        results = json.load(fp)

    # Quickly sums the times and states for the game and both players
    for game_version in results:
        sum_times_and_states(game_version)
        print_numbers_to_latex_table(game_version)

    # Saves these results back into the JSON file.
    with open('results/minimax_time_results.json', 'w+') as fp:
        json.dump(results, fp, indent=4)