import numpy as np
import matplotlib.pyplot as pyplot

def _plot_single_replication(axis, rewards, std=True, skip_every=1,
                        line_colour='tab:blue', std_colour='tab:orange', alpha=0.2):

    mean_rewards = np.mean(rewards, axis=0)
    skipped_mean_rewards = mean_rewards[::skip_every]

    episodes = np.linspace(1, len(mean_rewards), len(skipped_mean_rewards))
    
    [line] = axis.plot(episodes, skipped_mean_rewards, color=line_colour)

    if std:
        std_rewards = np.std(rewards, axis=0)
        skipped_std_rewards = std_rewards[::skip_every]

        axis.fill_between(
            episodes, 
            skipped_mean_rewards - skipped_std_rewards,
            skipped_mean_rewards + skipped_std_rewards,
            color=std_colour, 
            alpha=alpha)
        
    return line

def plot_single_replication(rewards, std=True, skip_every=1, x_limits=[0, 1000], y_limits=[0, 500],
                    labels=("Episodes", "Total Rewards"), title=None, image_size=(10, 5),
                    line_colour='tab:blue', std_colour='tab:orange', alpha=0.2):

    figure, axis = pyplot.subplots()
    figure.set_size_inches(image_size)

    _plot_single_replication(axis, rewards, std, skip_every, line_colour, std_colour, alpha)

    x_label, y_label = labels

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    if title is not None:
        axis.set_title(title)

    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)

    return figure, axis

def plot_multiple_replications(data, data_labels, std=True, skip_every=1, x_limits=[0, 1000], 
                    y_limits=[0, 500], axis_labels=("Episodes", "Total Rewards"), 
                    title=None, image_size=(10, 5), line_colours=None, std_colours=None, 
                    alpha=0.2, legend_location='upper left'):
    
    figure, axis = pyplot.subplots()
    figure.set_size_inches(image_size)

    line_colours = line_colours if line_colours is not None \
        else len(data) * ['tab:blue']

    std_colours = std_colours if std_colours is not None \
        else len(data) * ['tab:orange']

    # Package everything up for each dataset
    data_parameters = zip(data, data_labels, line_colours, std_colours)

    for (rewards, label, line_colour, std_colour) in data_parameters:
        line = _plot_single_replication(
            axis, rewards, std, skip_every, 
            line_colour, std_colour, alpha)
        
        line.set_label(label)
    
    x_label, y_label = axis_labels

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    if title is not None:
        axis.set_title(title)

    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)

    axis.legend(loc=legend_location)

    return figure, axis

def plot_doublings(x, y, labels, title=None, x_limits=None, y_limits=None, 
                image_size=(10, 5), colour="tab:blue"):
    
    figure, axis = pyplot.subplots()
    figure.set_size_inches(image_size)

    # This is a get-around for the bugginess of using log scale in matplotlib
    x_range = np.arange(len(x))
    axis.set_xticks(x_range)
    axis.set_xticklabels(list(map(str, x)))

    axis.plot(x_range, y, color=colour)

    x_label, y_label = labels

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    if title is not None:
        axis.set_title(title)

    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)

    return figure, axis