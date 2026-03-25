from abc import ABC, abstractmethod
import numpy as np, torch
import torch.nn as nn
import torch.nn.modules.loss as loss


class AOptimizer(ABC):

    def __init__(self, nn_model: nn.Module, loss_function):
        self._model = nn_model
        self._loss_function = loss_function

    @property
    def model(self) -> nn.Module:
        return self._model

    @model.setter
    def model(self, nn_model) -> None:
        self._model = nn_model

    @property
    def loss_function(self) -> loss._WeightedLoss:
        return self._loss_function

    @loss_function.setter
    def loss_function(self, loss_function: loss._WeightedLoss) -> None:
        self._loss_function = loss_function

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @abstractmethod
    def _optimize_impl(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    def optimize(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        print(f"Start optimization for {self.get_name()} Strategy")
        self._optimize_impl(X_train, y_train)
        acc = self.evaluate(X_test=X_train, y_test=y_train)
        print(f"Accuracy on Train Set: {acc:.2%}")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        preds = self._model(X_test).argmax(dim=1)
        acc = (preds == y_test).float().mean()
        return acc
