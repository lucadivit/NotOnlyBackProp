import numpy as np, torch

from optimizer.abstract.AOptimizer import AOptimizer

class CrystalOptimizer(AOptimizer):

    def __init__(self, nn_model, loss_function):
        super().__init__(nn_model=nn_model, loss_function=loss_function)

    @staticmethod
    def get_name() -> str:
        return "Crystal"

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

    def _assign_weights(self, crystal: np.ndarray) -> None:
        cumulative_nb_params = 0
        for p in self._model.parameters():
            total_params = p.numel()
            tensor = torch.tensor(crystal[cumulative_nb_params:cumulative_nb_params + total_params])
            reshaped_tensor = tensor.view(p.shape)
            p.data.copy_(reshaped_tensor)
            cumulative_nb_params += total_params

    def _evaluate_crystals(self, crystals: np.ndarray, X_train: torch.Tensor, y_train: torch.Tensor) -> (np.ndarray, float, int):
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

    def optimize(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        print(f"Start optimization for {CrystalOptimizer.get_name()} Strategy")

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)

        lower_bound, upper_bound = -2, 2
        nb_crystal = 15
        nb_iterations = 60
        crystals = self._create_crystals(lb=lower_bound, ub=upper_bound, nb_crystal=nb_crystal)
        fitnesses, best_fitness, best_index = self._evaluate_crystals(crystals=crystals, X_train=X_train_t, y_train=y_train_t)
        Cr_b = crystals[best_index]
        for i in range(0, nb_iterations):
            for crystal_idx in range(0, nb_crystal):
                new_crystals = np.array([])
                Cr_main = self._take_random_crystals(crystals=crystals, nb_random_crystals_to_take=1, nb_crystal=nb_crystal).flatten()
                Cr_old = crystals[crystal_idx]
                Fc = self._take_random_crystals(crystals=crystals, nb_crystal=nb_crystal).mean(axis=0)
                r, r_1, r_2, r_3 = self.__compute_r_values()
                Cr_new = self._compute_simple_cubicle(Cr_old=Cr_old, Cr_main=Cr_main, r=r)
                new_crystals = np.hstack((new_crystals, Cr_new))
                Cr_new = self._compute_cubicle_with_best_crystals(Cr_old=Cr_old, Cr_main=Cr_main, Cr_b=Cr_b, r_1=r_1, r_2=r_2)
                new_crystals = np.vstack((new_crystals, Cr_new))
                Cr_new = self._compute_cubicle_with_mean_crystals(Cr_old=Cr_old, Cr_main=Cr_main, Fc=Fc, r_1=r_1, r_2=r_2)
                new_crystals = np.vstack((new_crystals, Cr_new))
                Cr_new = self._compute_cubicle_with_best_and_mean_crystals(Cr_old=Cr_old, Cr_main=Cr_main, Cr_b=Cr_b, Fc=Fc, r_1=r_1, r_2=r_2, r_3=r_3)
                new_crystals = np.vstack((new_crystals, Cr_new))
                new_crystals = np.clip(new_crystals, a_min=lower_bound, a_max=upper_bound)
                new_crystal_fitnesses, new_crystal_best_fitness, new_crystal_best_index = self._evaluate_crystals(crystals=new_crystals, X_train=X_train_t, y_train=y_train_t)
                current_crystal_fitness = fitnesses[crystal_idx][0]
                if self._is_new_fitness_better(old_crystal_fitness=current_crystal_fitness, new_crystal_fitness=new_crystal_best_fitness):
                    crystals[crystal_idx] = new_crystals[new_crystal_best_index]

            fitnesses, best_fitness, best_index = self._evaluate_crystals(crystals=crystals, X_train=X_train_t, y_train=y_train_t)
            Cr_b = crystals[best_index]
            if i % 10 == 0:
                print(f"Iter {i}. Current Best Crystal Fitness Is {best_fitness}")
            self._assign_weights(crystal=Cr_b)
        acc = self.evaluate(X_test=X_train, y_test=y_train)
        print(f"Accuracy on Train Set: {acc:.2%}")
