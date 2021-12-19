import numpy as np
import json

import decision_tree as dt
import matplotlib.pyplot as plot

# Step 1: Load the dataset
dataset_filepath = 'files/clean_dataset.txt'
clean_dataset = np.loadtxt(dataset_filepath)

# Split into testing and training dataset (ratio = 0.2)
training_set, test_set = dt.dataset.create_train_and_test_sets(clean_dataset)

# Step 2: Train the decision tree
print(f"Training decision tree on data from {dataset_filepath}.")
tree, depth = dt.tree.decision_tree_learning(training_set)

# Step 4: Prune the decision tree
print("    - pruning.")
pruned_tree, pruned_depth = dt.prune.reduced_error_pruning(tree, test_set)

print(f"    - reduced max depth from {depth} to {pruned_depth}.")

tree_size = len(dt.prune.get_all_tree_items(tree))
pruned_tree_size = len(dt.prune.get_all_tree_items(pruned_tree))

print(f"    - reduced number of tree elements from {tree_size} to {pruned_tree_size}.")

print("    - done.\n")

# Step 3/5: Running evaluation and k-fold cross validation on the algorithm.
print("Running 10-fold cross validation on tree algorithm.")
results = dt.cross_validation.k_fold_cv(clean_dataset, 10, prune=True)

print("    - done.\n")

# Report the results
overall = results['overall']

matrix = np.array(overall['confusion_matrix'])
metrics = overall['metrics']

print("Confusion Matrix:")
print(matrix, "\n")

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1: {metrics['f1']:.3f}")

# Dump tree trained on clean dataset to JSON file
with open('tree_structure.json', 'w') as fp:
    json.dump(tree, fp, indent=4)

# Dump tree trained on clean dataset to JSON file
with open('pruned_tree_structure.json', 'w') as fp:
    json.dump(pruned_tree, fp, indent=4)

# Dump 10-fold cross validation results to JSON file
with open('tree_results.json', 'w') as fp:
    json.dump(results, fp, indent=4)

# BONUS: Create the tree image.
print("Generating tree images.")
tree_figure, tree_axis = dt.visualise.show(tree)
tree_figure.suptitle("Unpruned Tree Structure")

pruned_tree_figure, pruned_tree_axis = dt.visualise.show(pruned_tree)
pruned_tree_figure.suptitle("Pruned Tree Structure")

plot.show()

print("\nDone.")