import random
import numpy as np
from pymoo.core.sampling import Sampling as PymooSampling


class Sampling(PymooSampling):
    """Sampling class.

    """
    def __init__(self, gene_space: list[list[int]]):
        """Initialize the instance.

        Parameters
        ----------
        gene_space : list[list[int]]
            Description.

        Returns
        -------
        None
            Description.

        """
        super().__init__()
        self.gene_space = gene_space

    def _generate_sample(self):
        """Generate sample.

        """
        return np.array([random.choice(x) for x in self.gene_space])

    def _do(self, problem, n_samples, **kwargs):
        """Do.

        Parameters
        ----------
        problem : Any
            Description.
        n_samples : Any
            Description.
        **kwargs : dict
            Description.

        """
        samples = [self._generate_sample() for _ in range(n_samples)]
        return np.array(samples)
