from optimizer.abstract.AOptimizer import AOptimizer
import numpy as np

class Optimizer:

    def __init__(self, strategy: AOptimizer = None):
        self._strategy = strategy

    @property
    def strategy(self) -> AOptimizer:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: AOptimizer) -> None:
        self._strategy = strategy

    def optimize(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._strategy.optimize(X_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        return self._strategy.evaluate(X_test, y_test)

