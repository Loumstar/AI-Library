# Decision Tree Learning Coursework

This repository holds the code for the first __Intro to Machine Learning__ coursework, which focuses on developing a Decision Tree algorithm to predict which room a sensor is in based on the signal strength of multiple WiFi routers around a house.

## Description

`decision_tree/` is a python package which contains modules for training, testing, pruning and visualising a decision tree.

`decision_tree.py` is the main script used for running all functions in one file.

These modules are:

- `tree.py`, which handles the training and prediction aspects of the decision tree. The main functions of this module are `decision_tree_learning()`, which given a training dataset, returns a tree in the form of nested dictionaries, and `predict()`, which returns a set of label predictions given some input dataset.

- `testing.py`, which handles most of the testing and analysis. This module also includes `evaluate()`, which returns the accuracy of a trained decision tree.

- `cross_validation.py`, which contains a single function `k_fold_cv()` for running _k_-fold cross validation on a decision tree.

- `prune.py`, which focuses on pruning methods. The main function of this module is `reduced_error_pruning()` which reduces the number of tree elements, starting at the greatest depth and working its way up the tree until the accuracy stops improving.

- `visualise.py`, which contains functions for plotting the decision tree in matplotlib. The main function of this module is `show()`.

- `dataset.py`, which contains miscellaneous functions for splitting and modifying the datasets in training and testing.

## Usage

To run the main script while in this repository, use:

```bash
python decision_tree.py
```

This will showcase what the package does as required in the coursework.

### Specific functions

If you wish to use any specific function, import the function module or the whole package:

```python
import decision_tree as dt
from dt.test import evaluate
```

#### `dt.tree.decision_tree_learning()`

This function takes in a 2-dimensional numpy array, where each row is the set of values for a single instance. The function assumes the last value in each row is the classification label.

```python
tree, max_depth = dt.tree.decision_tree_learning(training_dataset)
```

Where `tree` is a series of nested dictionaries, starting from the root. Each dictionary represents a tree element (node or leaf).

#### `dt.tree.predict()`

This function also takes in a 2-dimensional numpy array, where each row is a single instance, and the trained tree.

```python
predictions = dt.tree.predict(instances, trained_tree)
```

Where `predictions` is a list of predicted labels.

#### `dt.test.evaluate()`

This function returns the accuracy of a trained tree, given a test dataset.

```python
accuracy = dt.test.evaluate(test_dataset, trained_tree)
```

Where `accuracy` is a float.

#### `dt.cross_validation.k_fold_cv()`

This function takes in a whole dataset, which is automatically shuffled and split into `k` train/test folds.

You also have the option to prune the tree before testing.

```python
results = dt.cross_validation.k_fold_cv(dataset, k, prune=False)
```

Where `results` is a dictionary containing the:

- Confusion matrix and metrics for each label in each fold.
- Macro-averaged metrics for each fold.
- Multi-label confusion matrix for each fold.
- The average metrics across all folds.
- The sum of multi-label confusion matrices across all folds.

#### `dt.prune.reduced_error_pruning()`

This function takes in a trained tree and a test dataset to reduce overfitting. Note, the evaluation is based off the __accuracy__ from the tests, rather than the accuracy.

```python
pruned_tree, max_depth = dt.prune.reduced_error_pruning(trained_tree, test_dataset)
```

Where pruned tree is an improved copy of the tree. Pruning does not alter the original tree.

#### `dt.visualise.show()`

This function takes in a tree and returns the figures/axis with the visualisation loaded.

```python
figure, axis = dt.visualise.show(trained_tree)
```

To see the tree, import `matplotlib` and run:

```python
plot.show()
```

## Repository Structure

```bash
decision_tree/ # Python package
    __init__.py
    tree.py
    test.py
    cross_validation.py
    prune.py
    visualise.py
    dataset.py

files/
    clean_dataset.txt
    noisy_dataset.txt

decision_tree.py # Main script for running all functions

tree_results.json # Record of decision tree performance during the cross-validation
tree_structure.json # The decision tree algorithm in JSON, which can be loaded back to dictionaries to be used again.
pruned_tree_structure.json # Same as above, but the pruned tree.

README.md
.gitignore
```
