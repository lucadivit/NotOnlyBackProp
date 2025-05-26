import numpy as np
import torch
import torch.nn as nn

from optimizer.abstract.AOptimizer import AOptimizer

class CrystalOptimizer(AOptimizer):

    def __init__(self, nn_model, loss_function):
        super().__init__(nn_model=nn_model, loss_function=loss_function)

    @staticmethod
    def get_name():
        return "Crystal"

    def _compute_fitnesses(self, crystals: np.array, eval_function: object) -> (np.array, float, int):
        fitnesses = np.apply_along_axis(eval_function, 1, crystals)
        best_fitness_index = np.argmin(fitnesses)
        best_fitness_value = fitnesses[best_fitness_index]
        fitnesses = fitnesses.reshape(-1, 1)
        return fitnesses, best_fitness_value, best_fitness_index

    def __compute_r_values(self, n: int = 2) -> (float, float, float):
        rnd_values = n * np.random.rand(4)
        rnd_values = np.round(rnd_values, decimals=5)
        r = round(rnd_values[0] - 1, 5)
        r_1 = round(rnd_values[1] - 1, 5)
        r_2 = round(rnd_values[2] - 1, 5)
        r_3 = round(rnd_values[3] - 1, 5)
        return r, r_1, r_2, r_3

    def _take_random_crystals(self, crystals: np.array, nb_crystal, nb_random_crystals_to_take: int = 0) -> np.array:
        if nb_random_crystals_to_take <= 0:
            nb_random_crystals_to_take = np.random.randint(low=1, high=nb_crystal + 1)
        indexes_of_random_crystal = np.random.choice(range(0, nb_crystal), nb_random_crystals_to_take, replace=False)
        return crystals[indexes_of_random_crystal]

    def _compute_simple_cubicle(self, Cr_old, Cr_main, r) -> np.array:
        return Cr_old + r * Cr_main

    def _compute_cubicle_with_best_crystals(self, Cr_old, Cr_main, Cr_b, r_1, r_2) -> np.array:
        return Cr_old + r_1 * Cr_main + r_2 * Cr_b

    def _compute_cubicle_with_mean_crystals(self, Cr_old, Cr_main, Fc, r_1, r_2) -> np.array:
        return Cr_old + r_1 * Cr_main + r_2 * Fc

    def _compute_cubicle_with_best_and_mean_crystals(self, Cr_old, Cr_main, Cr_b, Fc, r_1, r_2, r_3) -> np.array:
        return Cr_old + r_1 * Cr_main + r_2 * Cr_b + r_3 * Fc

    def _is_new_fitness_better(self, old_crystal_fitness, new_crystal_fitness) -> bool:
        return new_crystal_fitness < old_crystal_fitness

    def _flat_weights(self) -> np.ndarray:
        weights = []
        for p in self._model.parameters():
            weights.append(p.data.view(-1))
        weights = torch.cat(weights).detach().cpu().numpy()
        return weights

    def _create_crystals(self, lb: int, ub: int, nb_crystal: int) -> (np.array, int, int):
        base_weights = self._flat_weights()
        dim = base_weights.size
        random_crystals = np.random.uniform(low=lb, high=ub, size=(nb_crystal - 1, dim))
        crystals = np.vstack([base_weights, random_crystals])
        print(f"Created {crystals.shape[0]} Crystals With {crystals.shape[1]} Elements")
        return crystals

    def _assign_weights(self, crystal: np.ndarray):
        cumulative_nb_params = 0
        for p in self._model.parameters():
            total_params = p.numel()
            tensor = torch.tensor(crystal[cumulative_nb_params:cumulative_nb_params + total_params])
            reshaped_tensor = tensor.view(p.shape)
            p.data.copy_(reshaped_tensor)
            cumulative_nb_params += total_params

    def _evaluate_crystals(self, crystals: np.ndarray, X_train: torch.Tensor, y_train: torch.Tensor):
        losses = []
        for crystal in crystals:
            self._assign_weights(crystal)
            with torch.no_grad():
                outputs = self._model(X_train)
                loss = self._loss_function(outputs, y_train)

            losses.append(loss.item())

        fitnesses = np.array(losses).reshape(-1, 1)
        best_index = np.argmin(fitnesses)
        best_value = fitnesses[best_index, 0]
        return fitnesses, best_value, best_index



    def optimize(self, X_train: np.ndarray, y_train: np.ndarray):
        print(f"Start optimization for {CrystalOptimizer.get_name()} Strategy")

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        lower_bound, upper_bound = -2, 2
        nb_crystal = 3
        nb_iterations = 3
        crystals = self._create_crystals(lb=lower_bound, ub=upper_bound, nb_crystal=nb_crystal)
        fitnesses, best_fitness, best_index = self._evaluate_crystals(crystals=crystals, X_train=X_train, y_train=y_train)
        Cr_b = crystals[best_index]
        historical_loss = [best_fitness]
        historical_crystal = [list(Cr_b)]
        print(f"Current Best Crystal Loss: {best_fitness}")
        for _ in range(0, nb_iterations):
            pass
