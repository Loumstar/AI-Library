from .model import Model

import torch

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error

from torch.utils.data import TensorDataset, DataLoader

class Regressor():

    MODEL_PARAMETERS = [
        "input_size", "output_size", "nb_layers", "max_nb_nodes",
        "node_decay", "activation_type", "dropout_rate"
    ]

    def __init__(self, x, y=None, nb_epoch=500, batch_size=20, learning_rate=1e-3, weight_decay=0.01, 
                nb_layers=1, max_nb_nodes=10, node_decay=1.5, activation_type="LeakyReLU", dropout_rate=0, device='cpu'):

        self.device = torch.device(device)
        
        self.features_parameters = dict()
        self.target_parameters = dict()

        self.target_label = None

        X, _ = self._preprocessor(x, y, training=True)

        self.input_size = X.shape[1]
        self.output_size = 1

        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.nb_layers = nb_layers
        self.max_nb_nodes = max_nb_nodes

        self.node_decay = node_decay
        self.activation_type = activation_type
        self.dropout_rate = dropout_rate

        self.model = None

        self.loss_function = None
        self.optimiser = None

        self.loss_history = list()

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

    """ Helper functions for preprocessing data. """

    def _update_feature_minmax(self, column, name):
        self.features_parameters[name] = {
            "type": "normalise",
            "bounds": (min(column), max(column))
        }

    def _update_feature_encoding(self, column, name):
        self.features_parameters[name] = {
            "type": "encoded",
            "labels": list(set(column))
        }

    def __update_feature_parameters(self, column, name):
        if column.dtype == object:
            self._update_feature_encoding(column, name)
        else:
            self._update_feature_minmax(column, name)

    def _normalise(self, column, params):
        min_value, max_value = params['bounds']
        return (column - min_value) / (max_value - min_value)

    def _unnormalise(self, arr, name=None):
        if name is None:
            name = self.target_label

        params = self.features_parameters[name]
        min_value, max_value = params['bounds']

        return (arr * (max_value - min_value)) + min_value

    def _binarize(self, column, params):
        binarizer = LabelBinarizer()
        binarizer.fit(params['labels'])

        columns = binarizer.transform(column)

        return columns, binarizer.classes_

    def _add_column_to_dataframe(self, dataframe, column, name):
        dataframe[name] = column

    def _add_columns_to_dataframe(self, dataframe, arr, names):
        for i, name in enumerate(names):
            self._add_column_to_dataframe(dataframe, arr[:, i], name)

    def _normalise_feature(self, dataframe, column, name):
        parameters = self.features_parameters[name]

        if parameters['type'] == 'encoded':
            binarized_column, labels = self._binarize(column, parameters)
            self._add_columns_to_dataframe(dataframe, binarized_column, labels)
        else:
            column_normalised = self._normalise(column, parameters)
            self._add_column_to_dataframe(dataframe, column_normalised, name)

    def _fill_dataframe_with_mean(self, dataframe):
        means = dataframe.mean(numeric_only=True)
        return dataframe.fillna(means)

    def _preprocess_dataframe(self, x, training):
        x = self._fill_dataframe_with_mean(x)
        
        X = pd.DataFrame()

        for feature in x.columns:
            column = x[feature]

            if training:
                self.__update_feature_parameters(column, feature)
            
            self._normalise_feature(X, column, feature)

        return torch.tensor(X.values, dtype=torch.float).to(self.device)

    """ Helper functions for creating and training the Neural Network. """

    def _get_model_params(self):
        return {p:getattr(self, p) for p in Regressor.MODEL_PARAMETERS}

    def _init_model(self):
        model_params = self._get_model_params()
        self.model = Model(**model_params).to(self.device)

    def _init_loss_function(self):
        self.loss_function = torch.nn.MSELoss(reduction='sum')

    def _init_optimiser(self):
        self.optimiser = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    """ Main methods of the Regressor Class. """

    def _preprocessor(self, x, y=None, training=False):
        """ 
        Preprocess input of the network.
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """
        if training:
            self.features_parameters = dict()

            if isinstance(y, pd.Series):
                y = y.to_frame()
            if y is not None:
                [self.target_label] = y.columns

        X = self._preprocess_dataframe(x, training)
        Y = self._preprocess_dataframe(y, training) \
            if y is not None else None

        if training:
            self.input_size = X.shape[1]

        return X, Y

    def _run_epoch(self, loader, record_loss=False):
        losses = list()

        for x_batch, y_batch in loader:
            self.optimiser.zero_grad()
            
            # forward pass
            predictions = self.model(x_batch)
            loss = self.loss_function(predictions, y_batch)
            
            if record_loss:
                losses.append(float(loss))

            # backwards pass
            loss.backward()
            self.optimiser.step()

        return np.mean(losses) if record_loss else None

    def fit(self, x, y, record_loss=False):
        """
        Regressor training function
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).
        Returns:
            self {Regressor} -- Trained model.
        """
        if record_loss:
            self.loss_history = list()

        X, Y = self._preprocessor(x, y, training=True)

        self._init_model()
        self._init_optimiser()
        self._init_loss_function()

        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.nb_epoch):
            mean_loss = self._run_epoch(loader, record_loss=True)

            if record_loss:
                self.loss_history.append(mean_loss)

        return self

    def predict(self, x):
        X, _ = self._preprocessor(x)
        predictions = self._unnormalise(self.model(X))
        return predictions.detach().numpy()

    def rmse(self, x, y):
        return mean_squared_error(x, y, squared=False)

    def score(self, x, y):
        predictions = self.predict(x)
        return -self.rmse(y.to_numpy(), predictions)
