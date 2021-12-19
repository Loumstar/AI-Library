from matplotlib.patches import Rectangle
import matplotlib.pyplot as plot

from . import prune as dtp
from copy import deepcopy

def add_parent_reference(tree):
    """
    Method to add a pointer back to each element's parent.
    ---
    This is used in the visualisation script to help determine the position of tree 
    elements relative to its parent. This is also why a copy of the tree must be made, to
    avoid modifying the original.

    Args:
        - `tree` (dict): The tree structure to add back-references to.
    
    Returns:
        `None` (modfifications are done in-place)
    """
    for element in dtp.get_all_tree_items(tree):
        # If root element, set parent pointer to None
        if element['depth'] == 0:
            element['parent'] = None

        # Leaf has no children, so ignore all leaves
        if element['type'] == 'leaf':
            continue

        # Move into the children of the parent
        left_child = element['left']
        right_child = element['right']

        # Set the 'parent' attribute of each child to this element
        left_child['parent'] = element
        right_child['parent'] = element

def get_element_coords(element, root_coords, margins):
    """
    Method to get the coordinates of a tree element given its parent element coordinates.
    ---
    Args:
        - `element` (dict): The tree element to determine the coordinates of.
        - `root_coords` (tuple[float]): The coordinates of the root element in the tree.
        - `margins` (tuple[float]): The initial gap between adjacent elements 
            and the gap between different depths of elements respectively.
    
    Returns:
        - `x_coord` (float): The x coordinate of the tree element.
        - `y_coord` (float): The y coordinate of the tree element.
    """
    root_x_coord, root_y_coord = root_coords
    node_margin, depth_margin = margins

    # Get the parent element
    parent = element['parent']

    # Determine the spacing between the element and it's neighbour.
    margin = node_margin / (2.1 ** element['depth'])
    
    # If the root element, set the x coord to be the root_x_coord.
    if parent is None:
        x_coord = root_x_coord
    # Otherwise, set the x coord relative to its parent's coords
    else:
        parent_x_coord = parent['coords'][0]

        # If element is the left child of its parent, then shift element left 
        # relative to parent coord, else shift it right.

        x_coord = parent_x_coord - margin \
            if id(parent['left']) == id(element) \
            else parent_x_coord + margin

    # Set the y coord based solely on the element depth
    y_coord = root_y_coord - (depth_margin * element['depth'])

    return x_coord, y_coord

def draw_leaf(axis, element, box_size):
    """
    Method to draw the rectangle shape for a leaf tree element.
    ---
    Args:
        - `axis` (matplotlib.Axis): The axis instance to add the leaf rectangle to.
        - `element` (dict): The tree element to add the rectangle for. Must be a leaf.
        - `box_size` (tuple[float]): The width and height of the boxes representing 
            tree elements.

    Returns:
        `None`
    """
    box_width, box_height = box_size
    x_coord, y_coord = element['coords']

    caption = f"Leaf: {element['label']}"

    rectangle = Rectangle(
        (x_coord, y_coord), 
        box_width, 
        box_height, 
        edgecolor="green", 
        facecolor="white", 
        linewidth=4)

    axis.add_patch(rectangle)
    draw_caption(axis, element, caption, box_size)

def draw_node(axis, element, box_size):
    """
    Method to draw the rectangle shape for a node tree element.
    ---
    Args:
        - `axis` (matplotlib.Axis): The axis instance to add the node rectangle to.
        - `element` (dict): The tree element to add the rectangle for. Must be a node.
        - `box_size` (tuple[float]): The width and height of the boxes representing 
            tree elements.

    Returns:
        `None`
    """
    box_width, box_height = box_size
    x_coord, y_coord = element['coords']

    # Create the caption string.
    column, value = element['split']
    caption = f"A{column + 1} > {value}"

    # Create the rectangle element
    rectangle = Rectangle(
        (x_coord, y_coord), 
        box_width, 
        box_height, 
        edgecolor="blue", 
        facecolor="white", 
        linewidth=4)

    # Add rectangle and caption to axis
    axis.add_patch(rectangle)
    draw_caption(axis, element, caption, box_size)

def get_caption_coords(element, box_size):
    """
    Method to get the coordinates of a caption for a node/leaf rectangle.
    ---
    Args:
        - `element` (dict): The tree element to add the caption for.
        - `box_size` (tuple[float]): The width and height of the boxes representing 
            tree elements.
    
    Returns:
        - `x_coord` (float): The x coordinate of the caption element.
        - `y_coord` (float): The y coordinate of the caption element.
    """
    box_width, box_height = box_size

    # Get element coords
    x_coord, y_coord = element['coords']
    # Shift caption to center of box
    x_coord += box_width / 2
    # Place caption just above the box
    y_coord += 2 * box_height

    return x_coord, y_coord

def draw_caption(axis, element, caption, box_size):
    """
    Method to add caption to a tree element.
    ---
    Args:
        - `axis` (matplotlib.Axis): The axis instance to add the caption to.
        - `element` (dict): The tree element to add the caption for.
        - `caption` (str): The caption text to add.
        - `box_size` (tuple[float]): The width and height of the boxes representing 
            tree elements.

    Returns:
        `None`
    """
    # Get the coordinates of the caption
    caption_coords = get_caption_coords(element, box_size)
    
    # Add the caption to the axis
    axis.annotate(
        caption, 
        caption_coords,
        color='black',
        weight='bold',
        fontsize=5,
        ha='center',
        va='center')

def draw_branch(axis, element, box_size):
    """
    Method to plot the line representing a branch between two tree elements.
    ---
    Args:
        - `axis` (matplotlib.Axis): The axis instance to add branch lines to.
        - `element` (dict): The tree element to draw a branch to from its parent.
        - `box_size` (tuple[float]): The width and height of the boxes representing 
            tree elements.

    Returns:
        `None`
    """
    box_width, box_height = box_size
    parent = element['parent']
    
    # Calculate the starting location of branch line (at parent)
    start_x_coord = parent['coords'][0] + (box_width / 2)
    start_y_coord = parent['coords'][1] + (box_height / 2)

    # Calculate the ending location of the branch line (at element)
    end_x_coord = element['coords'][0] + (box_width / 2)
    end_y_coord = element['coords'][1] + (box_height / 2)

    # Plot line on axis
    axis.plot(
        [start_x_coord, end_x_coord], 
        [start_y_coord, end_y_coord],
        linewidth=1,
        marker=' ',
        c='r')

def show(tree, root_coords=(0, 300), box_size=(2, 2), margins=(300, 10)):
    """
    Method to create a figure that plots the decision tree.
    ---
    Args:
        - `tree` (dict): The tree to build a visualisation for.
        - `root_coords` (tuple[float], optional): The coordinates of the root tree in the graph.
        - `box_size` (tuple[float], optional): The width and height of the boxes 
            representing tree elements.
        - `margins` (tuple[float], optional): The initial gap between adjacent elements 
            and the gap between different depths of elements respectively.

    Returns:
        - `figure` (matplotlib.Figure): The figure instance of the visualisation.
        - `axis` (matplotlib.Axis): The axis instance of the visualisation,
    """
    # Copy the tree
    tree_copy = deepcopy(tree)
    # For each element, add links to its parent (for finding relative coords)
    add_parent_reference(tree_copy)

    # Create figure and axis instances
    fig, axis = plot.subplots()

    for element in dtp.get_all_tree_items(tree_copy):
        # Determine the coordinates of an element
        element['coords'] = get_element_coords(element, root_coords, margins)
        
        # Add rectangle and caption to figure
        if element['type'] == 'node':
            draw_node(axis, element, box_size)
        else:
            draw_leaf(axis, element, box_size)

        # Draw the line representing a branch between two elements
        if element['depth'] != 0:
            draw_branch(axis, element, box_size)

    # Remove axis scales in visualisation.
    plot.axis("off")

    return fig, axis