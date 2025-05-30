import random

import numpy as np
from pymoo.core.sampling import Sampling as PymooSampling


class Sampling(PymooSampling):
    def __init__(self, gene_space: list[list[int]]):
        super().__init__()
        self.gene_space = gene_space

    def _generate_sample(self):
        return np.array([random.choice(x) for x in self.gene_space])

    def _do(self, problem, n_samples, **kwargs):
        samples = [self._generate_sample() for _ in range(n_samples)]
        return np.array(samples)
