import numpy as np

def split_dataset(dataset, split):
    """
    Method to split a dataset into two based on a split.
    ---
    This is used during training to determine whether a split at a given feature/value
    in the dataset maximises the information gain.

    Args:
        - `dataset` (np.ndparray): The 2-dimensional dataset to split.
        - `split` (tuple[int, float]): The split point (feature, value) to separate the
            dataset by.

    Returns:
        - `left_dataset` (np.ndarray): The dataset that where the feature value was
            less than that of the split point.
        - `right_dataset` (np.ndarray): The dataset that where the feature value was
            greater than that of the split point.
    """
    column, value = split
    s_left, s_right = [], []

    for data in dataset:
        # If feature value for datapoint is less than the split value
        if data[column] < value:
            # Add to the left dataset
            s_left.append(data)
        else:
            # Else add to the right dataset
            s_right.append(data)

    # Convert datasets back to numpy arrays
    return np.array(s_left), np.array(s_right)

def create_train_and_test_sets(dataset, ratio=0.2):
    """
    Method to split a dataset up into a training and testing dataset.
    ---
    Args:
        - `dataset` (np.ndarray): The dataset to split.
        - `ratio` (float): The ratio of test datapoints to training datapoints. 
            By default, this is 0.2 (1:5 ratio).

    Returns:
        - `training_dataset` (np.ndarray): The training dataset.
        - `testing_dataset` (np.ndarray): The testing dataset.
    """
    # Randomly shuffle the dataset
    np.random.shuffle(dataset)

    test_length = int(len(dataset) * ratio)
    return dataset[test_length:], dataset[:test_length]


def split_dataset_by_k_folds(dataset, k):
    """
    Method to split a dataset up into k evenly sized folds.
    ---
    Args:
        - `dataset` (np.ndarray): The dataset to split.
        - `k` (int): The number of folds to split into.

    Returns:
        - `folds` (np.ndarray): A 3-dimensional array, 
            where each row is a separate fold.
    """
    np.random.shuffle(dataset)
    
    return np.vsplit(dataset, k)