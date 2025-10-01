import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import numpy as np
import optuna
import pandas as pd

from .constraints import Constraints
from .objective import Objective
from .variable_choosing import VariableChooser, WeightChooser
from .variable_ordering import NeutralOrder, VariablesOrder


@dataclass
class OptimizerMetrics:
    """
    Class to store and manage optimization metrics during the optimization process.

    This class keeps track of the best objective values, the evaluation status of the
    objective function, and the total number of function evaluations. It provides
    methods to update these metrics during the optimization run.

    Attributes
    ----------
    _best_values : List[float]
        A list to store the best objective values after each trial.
    _called_obj : List[bool]
        A list to indicate whether the objective function was evaluated in each trial.
    _func_evals_total : List[int]
        A list to keep track of the cumulative number of function evaluations.
    """

    _best_values: List[float] = field(default_factory=list, repr=False)
    _called_obj: List[bool] = field(default_factory=list, repr=False)
    _func_evals_total: List[int] = field(default_factory=list, repr=False)

    @property
    def best_values(self):
        """
        Get the list of best objective values recorded during optimization.

        Returns
        -------
        List[float]
            List of best objective values for each trial.
        """
        return self._best_values

    @property
    def called_obj(self):
        """
        Get the list indicating whether the objective was called in each trial.

        Returns
        -------
        List[bool]
            List of boolean values indicating objective function calls.
        """
        return self._called_obj

    @property
    def func_evals_total(self):
        """
        Get the list of cumulative function evaluation counts.

        Returns
        -------
        List[int]
            List of total function evaluations after each trial.
        """
        return self._func_evals_total

    def update_metrics(self, obj_value: float, called_obj: bool, func_evals: int) -> None:
        """
        Update the metrics with new values.

        Parameters
        ----------
        obj_value : float
            The value of the objective function.
        called_obj : bool
            Whether the objective function was evaluated.
        func_evals : int
            The total number of function evaluations.
        """
        if len(self._best_values) == 0:
            self._best_values.append(obj_value)
        else:
            self._best_values.append(max(obj_value, self._best_values[-1]))

        self._called_obj.append(called_obj)
        self._func_evals_total.append(func_evals)


class Optimizer(ABC):
    """
    Abstract base class for an optimizer. Defines the structure for optimization algorithms.
    """

    def __init__(self, objective: Objective, constraints: Constraints):
        """
        Initialize the optimizer with objective and constraints.

        Parameters
        ----------
        objective : Objective
            The objective function to optimize.
        constraints : Constraints
            The constraints that the solution must satisfy.
        """
        self._objective = objective
        self._constraints = constraints
        self._metrics: OptimizerMetrics = OptimizerMetrics()

    @property
    def metrics(self):
        """
        Get the optimizer metrics object tracking optimization progress.

        Returns
        -------
        OptimizerMetrics
            Object containing optimization performance metrics.
        """
        return self._metrics

    @abstractmethod
    def run(self, max_runs: int, timeout: float, verbose: bool) -> tuple[dict, float, float, int]:
        """
        Run the optimization process.

        Parameters
        ----------
        max_runs : int
            Maximum number of optimization runs/trials to execute.
        timeout : float
            Maximum time in seconds for the optimization process. If reached, the optimization will stop.
        verbose : bool
            Whether to print progress information during optimization.

        Returns
        -------
        tuple[dict, float, float, int]
            A tuple containing:
            - dict: The best solution found (variable assignments)
            - float: The objective value of the best solution
            - float: The time taken for the optimization in seconds
            - int: The number of function evaluations performed
        """
        pass

    @abstractmethod
    def get_best_x(self) -> dict:
        """
        Get the best solution found during optimization.

        Returns
        -------
        dict
            Dictionary mapping variable names to their optimal values.
        """
        pass


class TPEOptimizer(Optimizer):
    """
    Tree-structured Parzen Estimator (TPE) based optimizer.
    Uses the Optuna library for optimization.
    """

    def __init__(
        self,
        objective: Objective,
        constraints: Constraints,
        vars_order: VariablesOrder = None,
        vars_chooser: VariableChooser = None,
    ):
        """
        Initialize the TPE optimizer with objective, constraints, and variable ordering.

        Parameters
        ----------
        objective : Objective
            The objective function to optimize.
        constraints : Constraints
            The constraints that the solution must satisfy.
        vars_order : VariablesOrder, optional
            The order in which variables are optimized (default is NeutralOrder).
        vars_chooser : VariableChooser, optional
            Strategy for choosing variables to optimize (default is WeightChooser).
        """
        super().__init__(objective, constraints)

        # Initialize Optuna study with TPE sampler
        sampler = optuna.samplers.TPESampler(
            prior_weight=2,
            consider_endpoints=True,
            n_startup_trials=0,
            n_ei_candidates=200,
        )
        self._study = optuna.create_study(direction=optuna.study.StudyDirection.MAXIMIZE, sampler=sampler)

        # Set variable ordering method
        self._vars_order = NeutralOrder() if vars_order is None else vars_order

        self._vars_chooser = WeightChooser() if vars_chooser is None else vars_chooser

        # Set up logging for the optimization process
        logging.basicConfig(filename="OptunaOptimizer.log", level=logging.INFO, filemode="w")
        optuna.logging.set_verbosity(optuna.logging.INFO)
        self._logger = optuna.logging.get_logger("optuna")

    def _save_run_results(self):
        """
        Save the trial results and provisions data to CSV files for further analysis.

        This method generates two CSV files:
        - 'provisions.csv' containing the best objective values and corresponding provisions.
        - 'trials.csv' containing detailed trial data, such as parameters, objective values, and states.
        """
        provisions_data = [
            {
                "best_val": self.metrics.best_values[trial.number],  # Best value achieved in the trial
                "provisions": trial.user_attrs.get("provisions", None),  # Provisions for the trial
            }
            for trial in self._study.trials
            if trial.value is not None  # Only include valid trials
        ]

        # Save provisions data to CSV
        df = pd.DataFrame(provisions_data)
        df.to_csv("provisions.csv", index=False)
        logging.info("Provisions data has been saved to provisions.csv.")

        trials_data = [
            {
                "trial_number": trial.number,  # Trial number
                "value": trial.value if trial.value is not None else 0,  # Objective value for the trial
                "penalty": trial.value if trial.value is not None else 0,  # Penalty for the trial
                "state": trial.state.name,  # State of the trial (e.g., COMPLETE, PRUNED, FAIL)
                "best_val": self.metrics.best_values[trial.number],  # Best value for the trial
                "called_obj": self.metrics.called_obj[trial.number],  # Whether the objective was called
                "func_evals": self.metrics.func_evals_total[trial.number],  # Total function evaluations
                "params": trial.params,  # Parameters for the trial
            }
            for trial in self._study.trials
        ]

        # Save detailed trial data to CSV
        df = pd.DataFrame(trials_data)
        df.to_csv("trials.csv", index=False)
        logging.info("All trial data has been saved to trials.csv.")

    def get_last_trial(self) -> optuna.Trial | optuna.trial.FrozenTrial | None:
        """
        Get the last successful trial from the optimization study.

        Returns
        -------
        optuna.Trial | optuna.trial.FrozenTrial | None
            The last trial with a positive objective value, or None if no such trial exists.
        """
        last_trial = None
        for _trial in reversed(self._study.trials):
            trial_val = _trial.value if _trial.value is not None else 0
            if trial_val > 0:
                last_trial = _trial
                break
        return last_trial

    def _optuna_objective(self, trial: optuna.Trial):
        """
        The objective function used by Optuna during the optimization process.

        Parameters
        ----------
        trial : optuna.trial.Trial
            The current trial object in Optuna.

        Returns
        -------
        float
            The objective value for the trial.
        """

        def trial_callback(var_num, low, high):
            if trial.user_attrs.get("initial", False):
                val = trial.suggest_int(
                    name=f"x_{var_num}", low=0, high=1e9
                )  # workaround for no warning during initial run
            elif var_num in vars_opt:
                val = trial.suggest_int(name=f"x_{var_num}", low=low, high=high)
            elif var_num in vars_fixed:
                if last_trial_suggested:
                    val = trial.suggest_int(name=f"x_{var_num}", low=low, high=high)  # workaround
                else:  # truly fixed for gradient
                    last_trial = self.get_last_trial()
                    last_trial_val = last_trial.user_attrs.get("params")[var_num]
                    val = trial.suggest_int(name=f"x_{var_num}", low=last_trial_val, high=last_trial_val)
            else:
                val = trial.suggest_int(name=f"x_{var_num}", low=0, high=0)
            return val

        def trials_data_callback():
            n = self._objective.num_params
            second_last_trial = [self._objective(np.zeros(n))[1], np.zeros(n), self._objective(np.zeros(n))[0]]
            last_trial = []
            for _trial in reversed(self._study.trials):
                trial_val = _trial.value if _trial.value is not None else 0
                if trial_val > 0:
                    if len(last_trial) == 0:
                        last_trial = [
                            _trial.value,
                            _trial.user_attrs.get("params"),
                            _trial.user_attrs.get("provisions", None),
                        ]
                    else:
                        second_last_trial = [
                            _trial.value,
                            _trial.user_attrs.get("params"),
                            _trial.user_attrs.get("provisions", None),
                        ]
                        break

            if len(last_trial) == 0:
                last_trial = [self._objective(np.zeros(n))[1], np.zeros(n), self._objective(np.zeros(n))[0]]

            return second_last_trial, last_trial

        last_trial_suggested = (
            False if self.get_last_trial() is None else self.get_last_trial().user_attrs.get("suggested", True)
        )
        n = self._objective.num_params
        vars_order = np.arange(n, dtype=int)
        vars_opt = vars_order
        vars_fixed = []

        vars_order = self._vars_order(vars_order)
        if trial.number > 0:
            if last_trial_suggested:
                vars_opt = []
                vars_fixed = vars_order
            else:
                vars_opt, vars_fixed = self._vars_chooser(vars_order, trials_data_callback)

        x = self._constraints.suggest_solution(vars_order, trial_callback)

        value = 0
        if not self._constraints.check_constraints(x):
            value = -self._objective.get_penalty(x)
            self.metrics.update_metrics(value, False, self._objective.current_func_evals)
            logging.info(
                f"Trial {trial.number}: PRUNED -> Params: x={x}, Func evals: {self._objective.current_func_evals}"
            )
            raise optuna.TrialPruned()  # Stop trial if constraints are violated
        else:
            provisions, value = self._objective(x, not last_trial_suggested)
            trial.set_user_attr("provisions", provisions)
            trial.set_user_attr("suggested", not last_trial_suggested)
            correct_x = np.array([var.count for var in self._constraints.correct_X(x)])
            trial.set_user_attr("params", correct_x)
            self._constraints.update_solution(correct_x)
            logging.info(
                f"Trial {trial.number}: COMPLETE -> Params: x={correct_x}, Value: {value}, Func evals: {self._objective.current_func_evals}"
            )
            self.metrics.update_metrics(value, True, self._objective.current_func_evals)
        return value

    def _run_initial(self):
        """
        Run a single initial optimization trial to start the process.
        """
        n = self._objective.num_params
        vars_order = np.arange(n, dtype=int)

        vars = self._vars_order(vars_order)

        x = self._constraints.suggest_initial_solution(vars)

        self._study.enqueue_trial({f"x_{i}": x[i] for i in range(n)}, user_attrs={"initial": True})

    def get_best_x(self) -> dict:
        """
        Get the best solution parameters found during optimization.

        Returns
        -------
        dict
            Dictionary mapping variable names to their optimal values.
        """
        return self._study.best_params

    def _check_stop(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """
        Check if the optimization process should stop based on function evaluations.

        Parameters
        ----------
        study : optuna.Study
            The current Optuna study object.
        trial : optuna.trial.FrozenTrial
            The current trial being evaluated.
        """
        last_trial = self.get_last_trial()
        if last_trial is not None and last_trial.user_attrs.get("suggested", True):
            return  # Do not stop if the last trial was suggested
        if not self._objective.check_available_evals():
            study.stop()  # Stop optimization if max function evaluations are reached
            logging.info("Optimization stopped due to exceeding maximum function evaluations.")
        if not self._objective.check_optimize_need():
            study.stop()
            logging.info("Optimization stopped due to no need for further optimization.")

    def run(
        self,
        max_runs: int,
        timeout: float | None,
        initial_runs_num: int = 1,
        verbose: bool = True,
    ) -> tuple[dict, float, float, int]:
        """
        Run the optimization process with the specified parameters.

        Parameters
        ----------
        max_runs : int
            Maximum number of runs for optimization.
        timeout : float | None
            Timeout duration in seconds.
        initial_runs_num : int, optional
            Number of initial runs to perform (default is 1).
        verbose : bool, optional
            Whether to display progress (default is True).

        Returns
        -------
        tuple[dict, float, float, int]
            A tuple containing the best parameters, the best objective value,
            the percentage of successful trials, and the total number of function evaluations.
        """
        n_jobs = 1  # Number of parallel jobs (default is 1)

        # Perform initial optimization runs
        for _ in range(initial_runs_num):
            self._run_initial()

        # Start the optimization process
        self._study.optimize(
            func=self._optuna_objective,  # Objective function for optimization
            n_trials=max_runs,  # Maximum number of trials
            timeout=timeout,  # Timeout for optimization
            n_jobs=n_jobs,  # Number of parallel jobs
            gc_after_trial=False,  # Disable garbage collection after each trial
            show_progress_bar=verbose,  # Display progress bar if verbose is True
            callbacks=[self._check_stop],  # Callback to check when to stop optimization
        )

        # Save results if verbose mode is enabled
        if verbose:
            self._save_run_results()

        # Return optimization results
        return (
            self._study.best_params,  # Best parameters found
            self._study.best_value,  # Best objective value
            sum(self.metrics.called_obj) / len(self.metrics.called_obj),  # Success rate of trials
            self.metrics.func_evals_total[-1],  # Total number of function evaluations
        )
