# Neural Network Coursework

This repository holds the code for the second __Intro to Machine Learning__ coursework, which focuses on developing a Neural Network model to predict the median price of houses in California. Therefore, this Neural Network is a supervised regression model.

## Description

`neural_network/` is a python package that contains modules for the regressor, model and hyperparameter tuning.

`house_value_regression.py` is the main script used for running the regressor and model in one file.

These modules are:

- `regressor.py`, which contains the `Regressor()` class for preprocessing, model initialisation, prediction and scoring.

- `model.py`, which contains the `Model()` class which creates a neural network designed to predict the house prices. This is a child of PyTorch's `nn.Module` class.

- `tuning.py`, which contains the `BruteForceGridSearchCV()` class and the `RegressorHyperParameterTuning()` function used for hyperparameter tuning.

## Usage

To run the main script for the housing prices regressor, use:

```bash
python house_value_regression.py
```

This will load the housing prices dataset, instantiate a regressor, perform hyperparameter search and save the regressor with the best parameters to a pickle file.

## Specific Classes/Functions (Regressor only)

If you wish to use any specific function, import the function module or the whole package:

```python
import neural_network
from neural_network.tuning import RegressorHyperParameterTuning
```

### `Regressor()`

This is the main class of the whole package. This builds, trains and tests the regression model, as well as performing data preprocessing.

```python
# Initialise the regressor
regressor = neural_network.Regressor(x_train)
# Train the neural network
regressor.fit(x_train, y_train)
# Predict new values
predictions = regressor.predict(x_test)
```

The Regressor constructor can take the following parameters for configuring the model and training operation:

- `learning_rate`: The learning rate of the optimiser.

- `nb_epoch`: The number of epochs to perform when training the model.

- `batch_size`: The number of samples to collect the losses of before performing an optimsation step.

- `weight_decay`: The L2 penalty of the optimiser (to prevent overfitting of the model).

- `nb_layers`: The number of hidden layers in the neural network model. Each layer is an instance of `torch.nn.Linear`.

- `max_nb_nodes`: The maximum number of nodes in a hidden layer. See `node_decay` for more info.

- `node_decay`: The rate at which the number of nodes decrease across the layers.

    This is so that we can control how the number of nodes varies between each layer. The number
    of nodes for a hidden layer `i` is given by the geometric series:

    ```python
    nodes_i = max_nb_nodes / (node_decay ** i)
    ```

- `activation_type`: The activation function used after each linear module. Only one activation function can be used.

    | Activation Function         | Value         |
    |-----------------------------|---------------|
    | Rectified Linear Unit       | `"ReLU"`      |
    | Leaky Rectified Linear Unit | `"LeakyReLU"` |
    | Tanh                        | `"Tanh"`      |
    | Sigmoid                     | `"Sigmoid"`   |

- `dropout_rate`: The dropout probability used after each linear module to avoid overfitting.

- `device`: The name of the device to use for training (for example, `"cuda"`). By default this is `"cpu"`.

### `Regressor.fit()`

Train the regression model. All values are normalised to between `[0, 1]`, or binarized into multiple single-label columns if textural data.

```python
regressor.fit(x_train, y_train, record_loss=True)
```

Where `x` is a `Dataframe` object of all the input features and `y` is a `Dataframe` or `Series` object of target values (must be only one column). Returns `self`, but trains in-place.

if `record_loss` is True, then the average MSE of every epoch during training is saved to `self.loss_history`. By default, this is False.

### `Regressor.predict()`

Predict the `y` values corresponding to each sample in `x`.

```python
predictions = regressor.predict(x)
```

Where `x` is a `pd.Dataframe` of input features. Returns a numpy array.

### `Regressor.score()`

Returns the negative RMSE of predicted values.

```python
score = regressor.score(x, y)
```

Where `x` is a `Dataframe` object of all the input features and `y` is a `Dataframe` or `Series` object of target values (must be only one column). Returns a `float`.

### `RegressorHyperParameterTuning()`

Performs an exhuastive grid search of parameters on the model to determine the best parameter set.

```python
target_label = "median_house_pricing"

grid = {
    "learning_rate": [1e-1, 1e-2, 1e-3, 1e-4],
    "batch_size": [20, 50, 200],
    "nb_layers": [0, 1, 2],
    "max_nb_nodes": [10, 20, 30],
    "activation_type": ["ReLU", "LeakyReLU", "Tanh"],
}

best_params = RegressorHyperParameterTuning(regressor, dataset, target_label, grid, save_results="results.json", save_best="best_params.json")
```

Where:

- `target_label` is the name of the column in the dataset to predict.
- `save_results` is the filename to dump the results of the grid search to in JSON.
- `save_best` is the filename to dump the best parameters of the search to in JSON.

By default, these results are not saved.

## Repository Structure

```bash
neural_network/
    __init__.py
    model.py
    regressor.py
    tuning.py

house_value_regression.py
housing.csv # dataset

readme.md
```
