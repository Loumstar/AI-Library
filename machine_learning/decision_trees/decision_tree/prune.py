from collections import deque
from copy import deepcopy

from . import test as dtt

def get_all_tree_items(tree):
    """
    Method to return a list of tree elements.
    ---
    This method doesn't promise to sort the elements by depth. For a sorted list of tree
    elements, use `sort_tree_items_by_depth()`.

    Args:
        - `tree` (dict): The decision tree to get all the elements of.
    
    Returns:
        - `tree_elements` (list[dict]): A list of tree elements.
    """
    # Start with the root element
    items = [tree]

    # If a leaf, stop recursion.
    if is_leaf(tree):
        return items

    # Else, invoke self recursively on its children
    items += get_all_tree_items(tree['left'])
    items += get_all_tree_items(tree['right'])

    return items

def sort_tree_items_by_depth(tree, reverse=False):
    """
    Method to return a sorted list of tree elements from the nested tree structure.
    ---
    Args:
        - `tree` (dict): The decision tree to get all the elements of.
        - `reverse` (bool, optional): Whether to order elements max depth last (False)
            or first (True). By default, this is False.

    Returns:
        - `tree_elements` (list[dict]): A list of sorted tree elements.
    """
    items = get_all_tree_items(tree)
    return sorted(items, reverse=reverse, key=lambda i: i['depth'])
    
def is_leaf(item):
    """
    Method to check whether a tree element is a leaf.
    ---
    Args:
        - `item` (dict): The tree element to test.

    Returns:
        - (bool): Returns True if item is a leaf.
    """
    return isinstance(item, dict) \
        and item.get('type') == 'leaf'

def is_two_leaf_node(item):
    """
    Method to check a tree element is a node with two leaves as children.
    ---
    Args:
        - `item` (dict): The tree element to test.
    
    Returns:
        - (bool): True if the element is a two-leaf node.
    """
    return isinstance(item, dict) \
        and not is_leaf(item) \
        and is_leaf(item['left']) \
        and is_leaf(item['right'])

def majority_class(node):
    """
    Method to get the majority-class label of a node with two leaves.
    ---
    Args:
        - `node` (dict): The node to find the majority class for.
    
    Returns:
        - `label` (float): The label with the highest count from its children.
        - `count` (int): The number of datapoints in the dataset that reached the
            majority label leaf during training.
    """
    children = [node['left'], node['right']]
    # Find the child with the max count
    majority_leaf = max(children, key=lambda l: l['count'])
    # Return that child's label and count
    return majority_leaf['label'], majority_leaf['count']

def reduced_error_pruning(tree, test_dataset):
    """
    Method to prune a decision tree based on the reduced error algorithm.
    ---
    This method returns a copy of the tree. Pruning does not affect the original.

    Args:
        - `tree` (dict): The decision tree to prune.
        - `test_dataset` (np.ndarray): The dataset to test the performance of pruned 
            trees against.

    Returns:
        - `pruned_tree` (dict): The pruned version of the original decision tree.
        - `max_depth` (int): The maximum depth of the pruned tree,
    """
    # Copy the tree
    pruned_tree = deepcopy(tree)
    # Create a queue of elements (max depth first) to try removing
    queue = deque(sort_tree_items_by_depth(pruned_tree, reverse=True))
    
    # Run initial tests on the original tree for later comparison
    results = dtt.test_results(tree, test_dataset)
    metrics = results['macro_averaged']
    
    # While there are tree elements that have not been pruned
    while queue:
        # Get the element and remove it from the queue
        item = queue.popleft()
        # If it isn't a node or doesn't have two leaves, ignore it
        if not is_two_leaf_node(item):
            continue

        # Save the original element to revert back to if accuracy doesn't improve
        original_node = item.copy()

        # Determine the majority class from its leaves
        label, count = majority_class(item)
        # Set the node to a leaf with the majority class label
        item.update({
            'type': 'leaf',
            'label': label,
            'count': count
        })

        # Run tests with the newly made leaf
        pruned_results = dtt.test_results(pruned_tree, test_dataset)
        pruned_metrics = pruned_results['macro_averaged']

        # If the performance is less, then revert back to original
        if pruned_metrics['accuracy'] < metrics['accuracy']:
            item.update(original_node)
        # Otherwise, update results with these for later comparison.
        else:
            metrics = pruned_metrics

            # Remove now unnecessary data from element
            del item['left']
            del item['right']
            del item['split']
        
    # Find the deepest leaf in the pruned tree structure.
    deepest_leaf = sort_tree_items_by_depth(pruned_tree, reverse=True)[0]

    return pruned_tree, deepest_leaf['depth']