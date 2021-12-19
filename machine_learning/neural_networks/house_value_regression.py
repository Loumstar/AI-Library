import pickle

import pandas as pd

from neural_network import Regressor
from neural_network.tuning import RegressorHyperParameterSearch

""" Helper Functions for Saving/Loading Model """

def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    
    print("\nSaved model in part2_model.pickle\n")

def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    
    print("\nLoaded model in part2_model.pickle\n")
    
    return trained_model

""" Main """

if __name__ == "__main__":

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    training_data = data[:12000]
    testing_data = data[12000:15000]

    # Spliting input and output
    x_train = training_data.loc[:, data.columns != output_label]
    y_train = training_data.loc[:, output_label]

    x_test = testing_data.loc[:, data.columns != output_label]
    y_test = testing_data.loc[:, output_label]

    regressor = Regressor(x_train, nb_epoch=1000, device='cpu')

    grid_search_dataset = data[:5000]

    grid = {
        "learning_rate": [1e-1, 1e-2, 1e-3, 1e-4],
        "batch_size": [20, 50, 200],
        "nb_layers": [0, 1, 2],
        "max_nb_nodes": [10, 20, 30],
        "activation_type": ["ReLU", "LeakyReLU", "Tanh"]
    }

    print("Running exhaustive grid search.")
    best_params = RegressorHyperParameterSearch(regressor, grid_search_dataset, output_label, 
                    grid, save_results='grid_results.json', save_best="best_params.json")
    
    regressor.set_params(**best_params)
    print("Fitting model with best parameters.")
    regressor.fit(x_train, y_train)
    
    score = regressor.score(x_test, y_test)
    print(f"Average RMSE is {score:.5f}")
    
    save_regressor(regressor)