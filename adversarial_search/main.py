import itertools
import json
import os
import sys

from game import Game

# Simple function to see what params are currently being tested
def print_params(**params):
    for param, value in params.items():
        print(f"    - {param}: {value}")

results = list()

# Set of different params ranges to test
win_length_range = range(2, 5)
column_range = range(2, 6)
row_range = range(2, 5)

# Package all the combinations up into an iterable
board_versions = list(itertools.product(column_range, row_range, win_length_range))

# Create a null stream so the game output isn't printed while testing
no_stream = open(os.devnull, 'w')

# Open the results file, starting as an empty list of results
with open('results/minimax_time_results.json', 'w+') as fp:
    json.dump(results, fp, indent=4) 

for i, (m, n, k) in enumerate(board_versions):
    # Set the parameters
    parameters = {
        "columns": m,
        "rows": n,
        "win_length": k
    }
    
    print(f"Testing board version {i+1}/{len(board_versions)} ~ {100 * (i+1)/len(board_versions):.2f}%.")
    print_params(**parameters)

    # The game may throw an error because some values of k are bigger than m and n.
    # Therefore, catch the ValueErrors and ignore them in the results
    try:
        game = Game(**parameters, prune=False, user=False)
    except ValueError:
        print("    - Game version ignored.")
        continue

    # stdout to the null stream
    sys.stdout = no_stream
    # Play the game
    game.play()
    # Reset the stream to print out test info
    sys.stdout = sys.__stdout__

    # Add the results to the list
    results.append({
        "parameters": parameters,
        "min": game.min_action_history,
        "max": game.max_action_history
    })

    # Dump the results to JSON after every test, incase we need to stop the 
    # program early because it's taking too long
    with open('results/minimax_time_results.json', 'w+') as fp:
        json.dump(results, fp, indent=4)   
  
print("Done.")

# Close the null stream
no_stream.close()