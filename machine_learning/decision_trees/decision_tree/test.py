import numpy as np

from . import tree as t

def evaluate(test_dataset, tree):
    """
    Method to return the accuracy of a tree using a test dataset.
    ---
    Args:
        `test_dataset` (np.ndarray): The test dataset. Assumes the last value in each
            datapoint is the correct label.
        `tree` (dict): The trained tree to test.

    Returns:
        `accuracy` (float): The accuracy of the tree on the test dataset.
    """
    answers = 0

    for x in test_dataset:
        if t.predict_one(x, tree) == x[-1]:
            answers += 1

    accuracy = answers / len(test_dataset)

    return accuracy

def confusion_matrix(predictions, answers):
    """
    Method to return the confusion matrix for single/multiple label predictions.
    ---
    Args:
        - `predictions` (np.ndarray): The array of label predictions.
        - `answers` (np.ndarray): The array of correct labels.
    
    Returns:
        - `matrix` (np.ndarray): A dimensional matrix of values, where the column
            are the predicted labels and the rows are the correct labels.
    """
    # Find all labels in the set
    labels = np.unique(answers)
    # Construct a row and column for each label.
    matrix = np.zeros((len(labels), len(labels)))

    for i, prediction in enumerate(predictions):
        # Finds the column index of the predicted label
        col = np.argwhere(labels == prediction)
        # Finds the row index of the actual label
        row = np.argwhere(labels == answers[i])
        # Add entry to matrix
        matrix[row, col] += 1

    return matrix

def confusion_matrix_by_label(predictions, answers):
    """
    Method to return a dictionary of single-label confusion matrices
    for each label in the set of preditions.
    ---
    The confusion matrices are keyed by the label name.

    Args:
        - `predictions` (np.ndarray): The array of label predictions.
        - `answers` (np.ndarray): The array of correct labels.

    Returns:
        - `confusion_matrices` (dict[np.ndarray]): A dictionary of confusion 
            matrices, keyed by label name.
    """
    confusion_matrices = dict()
    labels = np.unique(answers)

    for label in labels:
        """
        Reduces values to single labels: 0 (positive) and 1 (negative).

        This is done so that when sorted, positives will be towards the top left 
        and negatives towards the bottom right of the confusion matrix.
        """
        single_label_predictions = (predictions != label).astype(int)
        single_label_answers = (answers != label).astype(int)

        # Get the confusion matrix for the single label
        matrix = confusion_matrix(single_label_predictions, single_label_answers)
        # Add to dictionary of confusion matrices
        confusion_matrices[str(label)] = matrix

    return confusion_matrices
    
def evaluation_metrics(matrix):
    """
    Method to return the evaluation metrics from a single-label confusion matrix.
    ---
    Args:
        - `matrix` (np.ndarray): A single-label confusion matrix (i.e. 2x2 size).
    
    Returns:
        - `metrics` (dict): A dictionary containing the:
            - Confusion Matrix (in nested list form so its JSON encodable).
            - Accuracy (float).
            - Precision (float).
            - Recall (float).
            - F1-Measure (float.)

    Note:
        Confusion matrix is structed as:
            - Predicted labels along the columns.
            - Actual labels along the rows.
    
        Therefore np.sum(matrix[0, :]) is all actual positives
            and np.sum(matrix[:, 0]) is all predicted positives

        Returns 0 for any metric that returns a NaN.
    """
    # accuracy = correctly labelled datapoints / all datapoints
    accuracy = (matrix[0, 0] + matrix[1, 1]) / np.sum(matrix)
    # precision = true positives / all predicted positives
    precision = matrix[0, 0] / np.sum(matrix[:, 0]) \
        if np.sum(matrix[:, 0]) != 0 else 0
    # recall = true positives / all actual positives
    recall = matrix[0, 0] / np.sum(matrix[0, :]) \
        if np.sum(matrix[0, :]) != 0 else 0
    # f1 = (2 x precision x recall) / (precision + recall)
    f1 = 2 * precision * recall / (precision + recall) \
        if (precision + recall) != 0 else 0
    
    return {
        'confusion_matrix': matrix.tolist(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_by_label(predictions, answers):
    """
    Method to get the confusion matrix and metrics for each label in predictions.
    ---
    Args:
        - `predictions` (np.ndarray): The array of label predictions.
        - `answers` (np.ndarray): The array of correct labels.
    
    Returns:
        - `label_results` (dict): Dictionary containing the matrix and metrics for
            each label, keyed by label name.
    """
    # Get the single-label confusion matrix for each label
    matrices = confusion_matrix_by_label(predictions, answers)

    label_results = dict()

    for label, matrix in matrices.items():
        # For each confusion matrix, add the metrics and save to label results
        label_results[str(label)] = evaluation_metrics(matrix)

    return label_results

def get_mean_key_value(key, arr):
    return np.mean([i.get(key) for i in arr])

def macro_averaged_results(label_results):
    """
    Method to get the macro-averaged evaluation metrics for all labels in a fold.
    ---
    Args:
        - `label_results` (dict): Dictionary containing the matrix and metrics for
            each label, keyed by label name.
    
    Returns:
        - `results` (dict): Dictionary containing the macro-averaged metrics.
    """
    # Remove label keys, no longer needed for averaging
    results = label_results.values()

    return {
        'accuracy': get_mean_key_value('accuracy', results),
        'precision': get_mean_key_value('precision', results),
        'recall': get_mean_key_value('recall', results),
        'f1': get_mean_key_value('f1', results)
    }

def test_results(tree, test_dataset):
    """
    Method to get the test results for a single fold.
    ---
    Args:
        - `tree` (dict): The trained tree to test.
        - `test_dataset` (np.ndarray): The dataset to use in testing. The last value of
            each datapoint is assumed to be the correct label.

    Returns:
        - `results` (dict): The dictionary containing the:
            - The multi-label confusion matrix from the fold.
            - The single-label confusion matrix and metrics for each label.
            - The macro-averaged results.
    """
    predictions = t.predict(test_dataset, tree)
    answers = test_dataset[:, -1]

    cm_overall = confusion_matrix(predictions, answers)
    label_results = evaluate_by_label(predictions, answers)
    ma_result = macro_averaged_results(label_results)

    return {
        "confusion_matrix": cm_overall.tolist(),
        "label_results": label_results,
        "macro_averaged": ma_result
    }

def sum_confusion_matrices(folds):
    """
    Method to sum the matrices from all of the folds.
    ---
    Args:
        - `folds` (list): A list of fold results.
    
    Returns:
        - `confusion_matrix` (np.ndarray): The sum of all multi-label confusion
            matrices across the folds.
    """
    return np.sum([f['confusion_matrix'] for f in folds], axis=0)

def get_mean_evaluation_metrics(folds):
    """
    Method to average the macro-averaged metrics from all folds.
    ---
    Args:
        - `folds` (list): A list of fold results.
    
    Returns:
        - `metrics` (dict): A dictionary containing averaged evaluation metrics.

    """
    ma_results = [f['macro_averaged'] for f in folds]

    return {
        'accuracy': get_mean_key_value('accuracy', ma_results),
        'precision': get_mean_key_value('precision', ma_results),
        'recall': get_mean_key_value('recall', ma_results),
        'f1': get_mean_key_value('f1', ma_results)
    }

def overall_results(folds):
    """
    Method to get the overall results from a k-fold cross validation.
    ---
    Args:
        - `folds` (list): A list of fold results.
    
    Returns:
        - `results` (dict): A dictionary containing the:
            - Sum of the multi-label confusion matrices across all folds.
            - The mean of the macro-averaged evaluation metrics.
    """
    matrix = sum_confusion_matrices(folds).tolist()
    metrics = get_mean_evaluation_metrics(folds)

    return {
        "confusion_matrix": matrix,
        "metrics": metrics
    }