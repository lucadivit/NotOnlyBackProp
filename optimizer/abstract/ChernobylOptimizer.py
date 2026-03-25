import numpy as np, torch

from optimizer.abstract.AOptimizer import AOptimizer

class ChernobylOptimizer(AOptimizer):

    def __init__(self, nn_model, loss_function):
        super().__init__(nn_model=nn_model, loss_function=loss_function)
        self._n_particles = 20
        self._n_iter = 150
        self._lb, self._ub = (-2, 2)
        self._dim = None
        seed = None
        self._np = np.random.default_rng(seed)

    @staticmethod
    def get_name() -> str:
        return "Chernobyl"

    def _is_better(self, a: float, b: float) -> bool:
        return a < b

    def _init_particles(self) -> np.ndarray:
        base_weights = self._flat_weights()
        self._dim = base_weights.size
        random_particles = self._np.uniform(self._lb, self._ub, size=(self._n_particles - 1, self._dim))
        particles = np.vstack([base_weights, random_particles])
        print(f"Created {particles.shape[0]} Particles With {particles.shape[1]} Elements")
        return particles

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self._lb, self._ub)

    def _update_best_particles(self, fitness: float, pos: np.ndarray, alpha_score: float, alpha_pos: np.ndarray,
                               beta_score: float, beta_pos: np.ndarray, gamma_score: float, gamma_pos: np.ndarray):

        is_alpha_better = self._is_better(fitness, alpha_score)
        is_beta_better = self._is_better(fitness, beta_score)
        is_gamma_better = self._is_better(fitness, gamma_score)

        if is_alpha_better is True:
            alpha_score, alpha_pos = fitness, pos.copy()

        if is_alpha_better is False and is_beta_better is True:
            beta_score, beta_pos = fitness, pos.copy()

        if is_alpha_better is False and is_beta_better is False and is_gamma_better is True:
            gamma_score, gamma_pos = fitness, pos.copy()

        return alpha_score, alpha_pos, beta_score, beta_pos, gamma_score, gamma_pos

    def _flat_weights(self) -> np.ndarray:
        weights = []
        for p in self._model.parameters():
            weights.append(p.data.view(-1))
        weights = torch.cat(weights).detach().cpu().numpy()
        return weights

    def _assign_weights(self, particle: np.ndarray) -> None:
        cumulative_nb_params = 0
        for p in self._model.parameters():
            total_params = p.numel()
            tensor = torch.tensor(particle[cumulative_nb_params:cumulative_nb_params + total_params])
            reshaped_tensor = tensor.view(p.shape)
            p.data.copy_(reshaped_tensor)
            cumulative_nb_params += total_params

    def _evaluate_particle(self, particle: np.ndarray, X_train: torch.Tensor, y_train: torch.Tensor) -> float:
        self._assign_weights(particle)
        with torch.no_grad():
            outputs = self._model(X_train)
            loss = self._loss_function(outputs, y_train)

        return float(loss.item())

    def _optimize_impl(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)

        particles = self._init_particles()

        alpha_pos = np.zeros(self._dim)
        beta_pos = np.zeros(self._dim)
        gamma_pos = np.zeros(self._dim)

        alpha_score = np.inf
        beta_score = np.inf
        gamma_score = np.inf

        for it in range(self._n_iter):
            for particle_i in range(self._n_particles):
                current_particle = self._clip(particles[particle_i])
                particles[particle_i] = current_particle
                fitness = self._evaluate_particle(particle=current_particle, X_train=X_train_t, y_train=y_train_t)

                (alpha_score, alpha_pos, beta_score,
                 beta_pos, gamma_score, gamma_pos) = self._update_best_particles(fitness, current_particle,
                                                                                 alpha_score, alpha_pos,
                                                                                 beta_score, beta_pos,
                                                                                 gamma_score, gamma_pos)

            walking_speed = 3.0 - (3.0 * it / self._n_iter)

            speed_alpha = np.log(self._np.uniform(1.0, 16000.0))
            speed_beta = np.log(self._np.uniform(1.0, 270000.0))
            speed_gamma = np.log(self._np.uniform(1.0, 300000.0))

            new_positions = np.empty_like(particles)

            for particle_i in range(self._n_particles):
                for particle_col in range(self._dim):

                    particle_col_val = particles[particle_i, particle_col]

                    r1 = self._np.random()
                    r2 = self._np.random()
                    alpha_prop = (np.pi * r1 * r1) / (0.25 * speed_alpha) - walking_speed * self._np.random()
                    alpha_prop_area = (r2 * r2) * np.pi
                    alpha_col_val = alpha_pos[particle_col]
                    D_alpha = abs(alpha_prop_area * alpha_col_val - particle_col_val)
                    gdf_alpha = 0.25 * (alpha_col_val - alpha_prop * D_alpha)

                    r1 = self._np.random()
                    r2 = self._np.random()
                    beta_prop = (np.pi * r1 * r1) / (0.5 * speed_beta) - walking_speed * self._np.random()
                    beta_prop_area = (r2 * r2) * np.pi
                    beta_col_val = beta_pos[particle_col]
                    D_beta = abs(beta_prop_area * beta_col_val - particle_col_val)
                    gdf_beta = 0.5 * (beta_col_val - beta_prop * D_beta)

                    r1 = self._np.random()
                    r2 = self._np.random()
                    gamma_prop = (np.pi * r1 * r1) / speed_gamma - walking_speed * self._np.random()
                    gamma_prop_area = (r2 * r2) * np.pi
                    gamma_col_val = gamma_pos[particle_col]
                    D_gamma = abs(gamma_prop_area * gamma_col_val - particle_col_val)
                    gdf_gamma = gamma_col_val - gamma_prop * D_gamma

                    new_positions[particle_i, particle_col] = (gdf_alpha + gdf_beta + gdf_gamma) / 3.0

            particles = self._clip(new_positions)
            if it % 10 == 0:
                print(f"Iter {it}. Current Best Particle Fitness Is {alpha_score}")
            self._assign_weights(particle=alpha_pos)
