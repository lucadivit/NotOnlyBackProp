import torch.optim as optim, torch, numpy as np
from torch.utils.data import TensorDataset, DataLoader

from optimizer.abstract.AOptimizer import AOptimizer


class BackpropOptimizer(AOptimizer):

    def __init__(self, nn_model, loss_function):
        super().__init__(nn_model=nn_model, loss_function=loss_function)

    @staticmethod
    def get_name():
        return "Backprop"

    def optimize(self, X_train: np.ndarray, y_train: np.ndarray) -> None:

        print(f"Start optimization for {BackpropOptimizer.get_name()} Strategy")

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), shuffle=False)

        opt = optim.Adam(self._model.parameters(), lr=0.01)
        for epoch in range(21):
            for xb, yb in train_loader:
                opt.zero_grad()
                out = self._model(xb)
                loss = self._loss_function(out, yb)
                loss.backward()
                opt.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        acc = self.evaluate(X_test=X_train, y_test=y_train)
        print(f"Accuracy on Train Set: {acc:.2%}")
