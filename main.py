import torch, torch.nn as nn, os, time
from copy import deepcopy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from SimpleNet import SimpleNet
from optimizer.Optimizer import Optimizer
from optimizer.abstract.BackpropOptimizer import BackpropOptimizer
from optimizer.abstract.CrystalOptimizer import CrystalOptimizer


def create_networks(weights: str = "weights.pth"):
    if os.path.exists(weights):
        print(f"Loading weights from {weights}")
        initial_weights = torch.load(weights)
    else:
        print("Creating and Saving new weights...")
        initial_weights = SimpleNet().state_dict()
        torch.save(initial_weights, weights)

    nn_model_backp = SimpleNet()
    nn_model_crystal = SimpleNet()
    nn_model_backp.load_state_dict(deepcopy(initial_weights))
    nn_model_crystal.load_state_dict(deepcopy(initial_weights))
    return nn_model_backp, nn_model_crystal

def get_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


nn_model_backp, nn_model_crystal = create_networks()
X_train, X_test, y_train, y_test = get_data()
loss = nn.CrossEntropyLoss()

backprop_opt = BackpropOptimizer(nn_model=nn_model_backp, loss_function=loss)
crystal_opt = CrystalOptimizer(nn_model=nn_model_crystal, loss_function=loss)
opt = Optimizer()

for concrete_strategy in [backprop_opt, crystal_opt]:
    start = time.perf_counter()

    opt.strategy = concrete_strategy

    acc = opt.evaluate(X_test=X_test, y_test=y_test)
    print(f"Accuracy Before Train with {concrete_strategy.get_name()}: {acc:.2%}")

    opt.optimize(X_train=X_train, y_train=y_train)
    acc = opt.evaluate(X_test=X_test, y_test=y_test)
    print(f"Accuracy After Train with {concrete_strategy.get_name()}: {acc:.2%}")

    end = time.perf_counter()
    elapsed = end - start
    print(f"Elapsed: {elapsed:.4f}s")