import math
import numpy as np
import random

__all__ = ["WhaleOptimizer"]


class WhaleOptimizer:
    def __init__(self, fitness_func, limit, population_n, sol_dim, seed, max_type=True):
        self._fitness_func = fitness_func
        self._limit = limit
        self._population_n = population_n
        self._sol_dim = sol_dim
        self._max_type = max_type
        np.random.seed(seed)
        self.rnd = random.Random(seed)

    def _init_population(self):
        self._population = np.random.uniform(
            self._limit[0], self._limit[1], size=(self._population_n, self._sol_dim)
        )

    def _get_best_sol_index(self):
        after_fitness = np.array(list(map(self._fitness_func, self._population)))
        return np.argmax(after_fitness) if self._max_type else np.argmin(after_fitness)

    def run(self, max_iter=20):
        self._init_population()
        min_bound = np.array([self._limit[0]] * self._sol_dim, dtype=np.float32)
        max_bound = np.array([self._limit[1]] * self._sol_dim, dtype=np.float32)
        for p in range(max_iter):
            x_star_index = self._get_best_sol_index()
            x_star = np.copy(self._population[x_star_index])
            s = 2 * (1 - p / max_iter)
            s2 = -1 + p * (-1 / max_iter)
            b = 1
            l = (s2 - 1) * self.rnd.random() + 1
            for idx, X in enumerate(self._population):
                V = self.rnd.random()
                K = 2 * s * V - s
                J = 2 * V
                t = self.rnd.random()
                B = abs(J * x_star - X)
                if t < 0.5:
                    if abs(K) < 1:
                        self._population[idx] = x_star - K * B
                    else:
                        option_lst = set([idx for idx in range(self._population_n)]) - {
                            p
                        }
                        xr = random.choice(list(option_lst))
                        self._population[idx] = xr - K * B
                else:
                    self._population[idx] = x_star + B * math.exp(b * l) * math.cos(
                        2 * math.pi * l
                    )
                # Make sure all value in bound range
                self._population[idx] = np.maximum(min_bound, self._population[idx])
                self._population[idx] = np.minimum(max_bound, self._population[idx])
        return np.rint(x_star).astype(int)

    