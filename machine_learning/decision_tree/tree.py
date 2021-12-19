import numpy as np

from . import dataset as ds

def entropy(dataset):
    """
    Method to get the information entropy of a dataset.
    ---
    Entropy is defined as sum(p_k * log2(p_k)) for probability p of choosing
    a label k in the dataset.

    Args:
        - `dataset` (np.ndarray): The daatset to find the information entropy of.

    Returns:
        - `entropy` (float): The information entropy.
    """
    # Count the number of times a label appears in the dataset
    _, label_counts = np.unique(dataset, return_counts=True)
    # Calculate the probability of randomly choosing each label
    p = label_counts / len(dataset)
    # Calculate information entropy for each probability
    return np.sum(-p * np.log2(p))

def information_gain(s, s_left, s_right):
    """
    Method to get the information gain of a split.
    ---
    Args:
        - `s` (np.ndarray): The whole dataset.
        - `s_left` (np.ndarray): The left-split of the dataset.
        - `s_right` (np.ndarray): The right-split of the dataset.
    """
    # remainder given by (entropy_left + entropy_right)
    entropy_left = len(s_left) / len(s) * entropy(s_left)
    entropy_right = len(s_right) / len(s) * entropy(s_right)

    # information gain given by total_entropy - remainer
    return entropy(s) - (entropy_left + entropy_right)

def find_split(dataset):
    """
    Method to return the split in the dataset that yields the greatest information gain.
    ---
    Args:
        - `dataset` (np.ndarray): This is the dataset to find the best split for.
            The last value of each datapoint is assumed to be the label.
    
    Returns:
        - `split` (tuple[int, float]): The split (column, value) that yields the greatest
            information gain.
    """
    _, n_inputs = dataset.shape

    best_split = (-1, -1)
    max_gain = 0

    # Loop through all columns except the labels column
    for i, col in enumerate(dataset.T[:n_inputs-1]):
        # Remove duplicates and sort values in column
        unique_col_values = np.unique(col)

        # For each potential split
        for value in unique_col_values:
            split = (i, value)
            # Split the dataset and determine the information gain
            s_left, s_right = ds.split_dataset(dataset, split)
            gain = information_gain(dataset, s_left, s_right)

            # If this information gain is the highest, save this split
            if gain > max_gain:
                best_split = split
                max_gain = gain

    # Return the best split
    return best_split

def decision_tree_learning(training_data, depth=0):
    """
    Method to train a decision tree from some training data.
    ---
    Args:
        - `training_data` (np.ndarray): A 2-dimensional array of datapoints used to 
            train the decision tree. The algorithm assumes the last value in each 
            datapoint is the label value.
        - `depth` (int, optional): The depth of the node to return a tree from, 
            since this function works recursively. By default, this is 0 as this would
            be the root element.

    Returns:
        - `tree` (dict): The trained tree, made up from a series of nested dictionaries
            where each one is a tree element (i.e. a node or a leaf).
        - `max_depth` (int): The maximum depth of the tree.
    """
    dataset_size, n_inputs = training_data.shape

    # Few debugging checks
    assert dataset_size != 0, "Empty dataset."
    assert n_inputs >= 2, "Not enough inputs."

    # Get all the labels in a single array
    labels = training_data[:,n_inputs-1]

    # If all labels in the set have the same value, return a leaf
    if np.all(labels == labels[0]):
        return {
            "type": "leaf",
            "label": labels[0],
            "depth": depth,
            "count": len(labels)
        }, depth

    # Find the split that results in the greatest information gain.
    split = find_split(training_data)

    assert split != (-1, -1), "Algorithm could not find a good split"

    # Separate the dataset according to this split.
    s_left, s_right = ds.split_dataset(training_data, split)

    # Train each branch on one of the datasets and repeat.
    left_branch, left_depth = decision_tree_learning(s_left, depth + 1)
    right_branch, right_depth = decision_tree_learning(s_right, depth + 1)

    # Return the root element and max depth
    return {
        "type": "node",
        "depth": depth,
        "split": split,
        "left": left_branch,
        "right": right_branch
    }, max(left_depth, right_depth)

def predict_one(x, tree):
    """
    Method to predict the label of a single instance using a decision tree.
    ---
    Args:
        - `x` (np.ndarray): A single instance to get the prediction of.
        - `tree` (dict): The trained tree used for prediction.

    Returns:
        - `label`: (float): The label the tree predicts the instance would have.
    """
    # If the root is a leaf, return its label
    if tree['type'] == 'leaf':
        return tree['label']
    # Find the split value
    col, value = tree['split']
    # Determine whether to traverse down the left or right branch.
    subtree = tree['left'] if x[col] < value \
        else tree['right']
    # Continue recusrively until a leaf is found.
    return predict_one(x, subtree)

def predict(instances, tree):
    """
    Method to return a list of predictions from the trained tree.
    ---
    Args:
        - `instances` (np.ndarray): A 2-dimensional array of datapoints to predict the
            labels of.
        - `tree` (dict): The trained tree to used for prediction.

    Returns:
        - `labels` (np.ndarray): A 1-dimensional array of predictions.
    """
    return np.array([predict_one(x, tree) for x in instances])
