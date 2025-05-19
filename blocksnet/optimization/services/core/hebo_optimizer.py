import logging

import pandas as pd
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

from .constraints import Constraints
from .objective import Objective
from .optimizer import Optimizer


class HEBOOptimizer(Optimizer):
    """
    HEBO-based optimizer for black-box function optimization.
    """

    def __init__(self, objective: Objective, constraints: Constraints):
        """
        Initialize the HEBO optimizer with an objective and constraints.

        Parameters
        ----------
        objective : Objective
            The objective function to optimize.
        constraints : Constraints
            The constraints that the solution must satisfy.
        """
        super().__init__(objective, constraints)

        # Define the search space based on the objective function
        space_config = [
            {"name": f"x_{i}", "type": "int", "lb": 0, "ub": self._constraints.get_ub(i)}
            for i in range(0, self._objective.num_params)
        ]

        self._design_space = DesignSpace().parse(space_config)
        self._optimizer = HEBO(self._design_space)

        # Setup logging
        logging.basicConfig(filename="HEBOOptimizer.log", level=logging.INFO, filemode="w")
        self._logger = logging.getLogger("hebo")

    def _evaluate(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate the objective function for given parameter values.

        Parameters
        ----------
        X : pd.DataFrame
            A dataframe containing parameter values to be evaluated.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the corresponding objective values.
        """
        results = []
        for _, row in X.iterrows():
            x = row.values.astype(int)
            if not self._constraints.check_constraints(x):
                value = self._objective.get_penalty(x)
                self.metrics.update_metrics(value, False, 0)
                results.append([-value])
                logging.info(f"Pruned: x={x}, value={value}")
            else:
                provisions, value = self._objective(x)
                self.metrics.update_metrics(value, True, self._objective.current_func_evals)
                results.append([-value])
                logging.info(f"Evaluated: x={x}, value={value}")
        return pd.DataFrame(results, columns=["fval"])

    def run(self, max_runs: int, timeout: float, verbose: bool = True) -> tuple[dict, float, float, int]:
        """
        Run the HEBO optimization process.

        Parameters
        ----------
        max_runs : int
            Maximum number of optimization runs.
        timeout : float
            Timeout in seconds for the optimization.
        verbose : bool, optional
            Whether to print progress during optimization (default is True).

        Returns
        -------
        tuple[dict, float, float, int]
            A tuple containing the best parameters, the best objective value,
            the percentage of successful trials, and the total number of function evaluations.
        """
        for n_run in range(max_runs):
            rec = self._optimizer.suggest(n_suggestions=2)  # TODO: вот здесь можно делать отбор решений

            result = self._evaluate(rec)
            self._optimizer.observe(rec, result.to_numpy())
            print("After %d iterations, best obj is %.2f" % (n_run, -self._optimizer.y.max()))

        best_value = -self._optimizer.y.max()
        success_rate = sum(self.metrics.called_obj) / len(self.metrics.called_obj)
        total_evals = self.metrics.func_evals_total[-1]

        return best_value, success_rate, total_evals
