import numpy as np
import json

from itertools import product
from sklearn.utils import shuffle

class BruteForceGridSearchCV():
    def __init__(self, estimator, target_label, parameter_grid):
        self.estimator = estimator
        self.target_label = target_label
        self.parameter_grid = parameter_grid

        self.keys, self.sets = self.get_parameter_sets()

        self.k = len(self.sets)

        self.results = list()

    def get_parameter_sets(self):
        keys = self.parameter_grid.keys()
        sets = set(product(*self.parameter_grid.values()))
        return keys, sets

    def get_k_folds(self, dataset, k):
        return np.array_split(dataset, k)

    def get_test_train_sets(self, df, folds, i):
        test_set = folds[i]
        train_set = df.drop(index=test_set.index)

        return train_set, test_set

    def get_mean_key_value(self, key, arr):
        return np.mean([i.get(key) for i in arr])

    def split_x_from_y(self, dataset):
        y = dataset[self.target_label]
        x = dataset.drop(columns=self.target_label)

        return x, y

    def best_fold(self):
        return max(self.results, key=lambda x: x.get("score", -np.inf))

    def print_parameter_set(self, params):
        for param, value in params.items():
            print(f"    - {param}: {value}")

    def save_result(self, result, params, folder=None):
        self.results.append(result)

        if folder is None:
            return

        params_strings = [f"{k}_{v}" for (k, v) in params.items()]
        filename = "__".join(params_strings) + ".json"

        with open(f"{folder}/{filename}", 'w') as f:
            json.dump(result, f, indent=4)

    def fit(self, dataset, reshuffle=True):
        print(f"Beginning Grid Search ({self.k} permutations).")
        self.results = list()
        
        if reshuffle:
            shuffle(dataset)

        folds = self.get_k_folds(dataset, self.k)

        for i, parameter_set in enumerate(self.sets):
            params = dict(zip(self.keys, parameter_set))

            print(f"Testing new parameter set ({i+1}/{self.k} ~ {100 * (i+1) / self.k:.1f}%):")
            self.print_parameter_set(params)

            train_set, validation_set = self.get_test_train_sets(dataset, folds, i)
            
            score = self.evaluate_param_set(train_set, validation_set, params)
            losses = self.estimator.loss_history

            result = {
                "params": params,
                "score": score,
                "losses": losses
            }

            self.save_result(result, params)
        
        print("Done.")
     
    def evaluate_param_set(self, train_set, test_set, params):
        x_train, y_train = self.split_x_from_y(train_set)
        x_test, y_test = self.split_x_from_y(test_set)

        self.estimator.set_params(**params)
        self.estimator.fit(x_train, y_train, record_loss=True)

        return self.estimator.score(x_test, y_test)

def RegressorHyperParameterSearch(estimator, dataset, target_label, parameter_grid,
                                    save_results=None, save_best=None): 
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-params. 

    """
    search = BruteForceGridSearchCV(estimator, target_label, parameter_grid)
    search.fit(dataset)

    best_parameters = search.best_fold().get("params")

    print("\nBest parameter set found:")
    search.print_parameter_set(best_parameters)

    if save_results is not None:
        with open(save_results, 'w') as f:
            json.dump(search.results, f, indent=4)

    if save_best is not None:
        with open(save_best, 'w') as f:
            json.dump(best_parameters, f, indent=4)

    return best_parameters