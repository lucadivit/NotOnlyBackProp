from optimizer.abstract.AOptimizer import AOptimizer
import numpy as np

class Optimizer:

    def __init__(self, optimizer: AOptimizer):
        self._optimizer = optimizer

    @property
    def optimizer(self) -> AOptimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: AOptimizer):
        self._optimizer = optimizer

    def optimize(self, X_train: np.ndarray, y_train: np.ndarray):
        self._optimizer.optimize(X_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        return self._optimizer.evaluate(X_test, y_test)

