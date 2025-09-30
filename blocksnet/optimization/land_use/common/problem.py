import networkx as nx
import numpy as np
from pymoo.core.problem import Problem as PymooProblem


class Problem(PymooProblem):
    """Problem class.

    """
    def __init__(self, n_var: int, n_classes: int, objectives: list):
        """Initialize the instance.

        Parameters
        ----------
        n_var : int
            Description.
        n_classes : int
            Description.
        objectives : list
            Description.

        Returns
        -------
        None
            Description.

        """
        self.objectives = objectives
        super().__init__(n_var=n_var, n_obj=len(objectives), n_constr=0, xl=0, xu=n_classes - 1, type_var=int)

    def _evaluate(self, solutions, out, *args, **kwargs):

        """Evaluate.

        Parameters
        ----------
        solutions : Any
            Description.
        out : Any
            Description.
        *args : tuple
            Description.
        **kwargs : dict
            Description.

        """
        solutions_objectives = [[] for _ in self.objectives]

        for solution in solutions:

            solution_objectives = [objective(solution) for objective in self.objectives]

            for i, value in enumerate(solution_objectives):
                solutions_objectives[i].append(value)

        out["F"] = np.column_stack([v for v in solutions_objectives])
