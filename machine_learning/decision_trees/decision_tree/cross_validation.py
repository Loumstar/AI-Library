import numpy as np

from . import tree as t
from . import dataset as ds
from . import test as dtt
from . import prune as dtp

def k_fold_cv(dataset, k, prune=False):
    """
    Method to run k-fold cross validation on a decision tree algorithm.
    ---
    Args:
        - `dataset` (np.ndarray): A 2-dimensional array of datapoints used for 
            training and testing.
        - `k` (int): The number of folds to run cross validation on.
        - `prune` (bool, optional): Choose whether to prune the tree before testing each
            fold. By default, this is False.
    
    Returns:
        - `results` (dict): A dictionary of results, including the:
            - Confusion matrix and metrics for each label in each fold.
            - Macro-averaged metrics for each fold.
            - Multi-label confusion matrix for each fold.
            - The average metrics across all folds.
            - The sum of multi-label confusion matrices across all folds.
    """
    results = {'folds': list(), 'overall': dict()}
    # Shuffle and split dataset into k folds
    folds = ds.split_dataset_by_k_folds(dataset, k)
    
    # For each test fold
    for i, fold in enumerate(folds):
        print(f"    - processing fold {i + 1}.")
        # Training set is the dataset minus the test fold
        training_folds = np.delete(folds, i, axis=0)
        training_dataset = np.concatenate(training_folds)

        print("         - training.")
        tree, _ = t.decision_tree_learning(training_dataset)

        if prune:
            print("         - pruning.")
            tree, _ = dtp.reduced_error_pruning(tree, fold)

        print("         - testing.")

        # Return the test results for that specific fold
        results['folds'].append(dtt.test_results(tree, fold))
        
    # Calculate the overall results from the metrics in each fold
    results['overall'] = dtt.overall_results(results['folds'])

    return results