from abc import ABC, abstractmethod
import numpy as np, torch


class AOptimizer(ABC):

    def __init__(self, nn_model, loss_function):
        self._model = nn_model
        self._loss_function = loss_function

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, nn_model):
        self._model = nn_model

    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, loss_function):
        self._loss_function = loss_function

    @abstractmethod
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_name():
        raise NotImplementedError

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        preds = self._model(X_test).argmax(dim=1)
        acc = (preds == y_test).float().mean()
        return acc
