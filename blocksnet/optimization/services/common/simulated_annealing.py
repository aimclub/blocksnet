import math
import random
from typing import Callable
from pydantic import BaseModel, Field
from tqdm import tqdm
from ....config import log_config


class SimulatedAnnealing(BaseModel):

    t_max: float = Field(ge=0)
    t_min: float = Field(ge=0)
    rate: float = Field(ge=0)
    iterations: int = Field(ge=0)

    @staticmethod
    def _perturbate(X: list[int]) -> tuple[list[int], int]:
        X = X.copy()
        i = random.choice(range(len(X)))
        delta = random.choice([-1, 1])
        X[i] = X[i] + delta
        return X, i

    def run(self, X: list[int], objective: Callable, constraint: Callable):

        best_X = X.copy()
        best_value = objective(best_X, None)

        curr_X = best_X.copy()
        curr_value = best_value

        T = self.t_max

        if not log_config.disable_tqdm:
            pbar = tqdm(range(self.iterations))

        for _ in range(self.iterations):

            if not log_config.disable_tqdm:
                pbar.update(1)
                pbar.set_description(f"obj : {curr_value : .2f}")

            X, i = self._perturbate(curr_X)

            if not constraint(X):
                continue

            value = objective(X, i)

            # remember best possible value
            if value > best_value:
                best_X = X
                best_value = value

            # simulate annealing
            if value > curr_value:
                curr_value = value
                curr_X = X
            else:
                delta = value - curr_value
                if random.random() < math.exp(delta / T):
                    curr_value = value
                    curr_X = X

            # cool
            T = T * self.rate
            if T < self.t_min:
                break

        return best_X, best_value
