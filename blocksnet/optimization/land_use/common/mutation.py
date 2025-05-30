import random

import numpy as np
from pymoo.core.mutation import Mutation as PymooMutation


class Mutation(PymooMutation):
    def __init__(self, probability: float, gene_space: list[list[int]]):
        super().__init__()
        self.probability = probability
        self.gene_space = gene_space

    def _mutate_gene(self, i, value):
        if random.random() <= self.probability:
            gene_space = self.gene_space[i]
            value = random.choice(gene_space)
        return value

    def _mutate_solution(self, solution):
        return [self._mutate_gene(i, v) for i, v in enumerate(solution)]

    def _do(self, problem, X, **kwargs):
        return np.array([self._mutate_solution(solution) for solution in X])
