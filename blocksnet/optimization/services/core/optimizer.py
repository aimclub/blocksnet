from blocksnet.config import log_config
from loguru import logger
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
        """Get the list of best objective values."""
        return self._best_values

    @property
    def called_obj(self):
        """Get the list indicating whether objective was called in each trial."""
        return self._called_obj

    @property
    def func_evals_total(self):
        """Get the list of total function evaluations."""
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
        """Get the optimizer metrics object."""
        return self._metrics

    @abstractmethod
    def run(self, max_runs: int, timeout: float) -> tuple[dict, float, float, int]:
        """
        Run the optimization process.

        Parameters
        ----------
        max_runs : int
            Maximum number of optimization runs/trials to execute.
        timeout : float
            Maximum time in seconds for the optimization process. If reached, the optimization will stop.

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
            The strategy for choosing variables (default is WeightChooser).
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
        log_config.set_logger_level(log_config.logger_level)
        
        optuna_level = getattr(optuna.logging, log_config.logger_level, optuna.logging.INFO)
        optuna.logging.set_verbosity(optuna_level)

    def _save_run_results(self):
        """
        Save the trial results and provisions data to CSV files for further analysis.

        This method generates two CSV files:
        - 'tpe_provisions.csv' containing the best objective values and corresponding provisions.
        - 'tpe_trials.csv' containing detailed trial data, such as parameters, objective values, and states.
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
        df.to_csv("tpe_provisions.csv", index=False)
        logger.info("Provisions data has been saved to tpe_provisions.csv.")

        trials_data = [
            {
                "trial_number": trial.number,  # Trial number
                "params": trial.params,
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
        df.to_csv("tpe_trials.csv", index=False)
        logger.info("All trial data has been saved to tpe_trials.csv.")

    def get_last_trial(self) -> optuna.Trial | optuna.trial.FrozenTrial | None:
        """
        Get the last successful trial from the study.

        Returns
        -------
        optuna.Trial | optuna.trial.FrozenTrial | None
            The last successful trial object, or None if no successful trials exist.
        """
        last_trial = None
        for _trial in reversed(self._study.trials):
            trial_val = _trial.value if _trial.value is not None else 0
            if trial_val > 0:
                if last_trial is None:
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
            """Callback function for suggesting variable values during optimization."""
            if var_num in vars_opt:
                val = trial.suggest_int(name=f"x_{var_num}", low=low, high=high)
            elif var_num in vars_fixed:
                last_trial = self.get_last_trial()
                val = trial.suggest_int(
                    name=f"x_{var_num}",
                    low=last_trial.user_attrs.get("params")[var_num],
                    high=last_trial.user_attrs.get("params")[var_num],
                )
            else:
                val = trial.suggest_int(name=f"x_{var_num}", low=0, high=0)
            return val

        def trial_callback_fixed(var_num):
            """Callback function for fixed variable values."""
            last_trial = []
            for i in range(len(self._study.trials) - 1, -1, -1):
                trial_val = self._study.trials[i].value if self._study.trials[i].value is not None else 0
                if trial_val > 0:
                    if len(last_trial) == 0:
                        last_trial = self._study.trials[i].user_attrs.get("params", [])
                        break
            val = trial.suggest_int(name=f"x_{var_num}", low=last_trial[var_num], high=last_trial[var_num])
            return val

        def trial_callback_null(var_num):
            """Callback function returning zero for unused variables."""
            return 0

        def trials_data_callback():
            """Callback function to get data from previous trials."""
            n = self._objective.num_params
            second_last_trial = [self._objective(np.zeros(n))[1], np.zeros(n)]
            last_trial = []
            for i in range(len(self._study.trials) - 1, -1, -1):
                trial_val = self._study.trials[i].value if self._study.trials[i].value is not None else 0
                if trial_val > 0:
                    if len(last_trial) == 0:
                        x = np.zeros(n)
                        for var, val in self._study.trials[i].params.items():
                            x[int(var[2:])] = val
                        last_trial = [self._study.trials[i].value, x]
                    else:
                        x = np.zeros(n)
                        for var, val in self._study.trials[i].params.items():
                            x[int(var[2:])] = val
                        second_last_trial = [self._study.trials[i].value, x]
                        break

            if len(last_trial) == 0:
                last_trial = [self._objective(np.zeros(n)), np.zeros(n)]
            else:
                last_trial[1] = last_trial[1]

            return second_last_trial, last_trial

        last_trial_suggested = (
            False if self.get_last_trial() is None else self.get_last_trial().user_attrs.get("suggested", True)
        )
        n = self._objective.num_params
        vars_order = np.arange(n, dtype=int)
        vars_opt = vars_order
        vars_fixed = []

        self._vars_order(vars_order)
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
            logger.info(
                f"Trial {trial.number}: PRUNED -> Params: x={x}, Func evals: {self._objective.current_func_evals}"
            )
            raise optuna.TrialPruned()  # Stop trial if constraints are violated
        else:
            provisions, value = self._objective(x)
            trial.set_user_attr("provisions", provisions)
            trial.set_user_attr("suggested", not last_trial_suggested)
            trial.set_user_attr("params", [var.count for var in self._constraints.correct_X(x)])
            logger.info(
                f"Trial {trial.number}: COMPLETE -> Params: x={x}, Value: {value}, Func evals: {self._objective.current_func_evals}"
            )
            self.metrics.update_metrics(value, True, self._objective.current_func_evals)
        return value

    def _run_initial(self):
        """
        Run a single initial optimization trial to start the process.
        """
        n = self._objective.num_params
        vars_order = np.arange(n, dtype=int)

        self._vars_order(vars_order)
        vars = vars_order

        x = self._constraints.suggest_initial_solution(vars)

        self._study.enqueue_trial({f"x_{i}": x[i] for i in range(n)})

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
        if not self._objective.check_available_evals() or not self._objective.check_optimize_need():
            study.stop()  # Stop optimization if max function evaluations are reached
            logger.info("Optimization stopped due to exceeding maximum function evaluations.")

    def run(
        self,
        max_runs: int,
        timeout: float | None,
        initial_runs_num: int = 1,
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

        Returns
        -------
        tuple[dict, float, float, int]
            A tuple containing:
            - dict: Best parameters found
            - float: Best objective value
            - float: Percentage of successful trials
            - int: Total number of function evaluations
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
            show_progress_bar=log_config.disable_tqdm,  # Display progress bar
            callbacks=[self._check_stop],  # Callback to check when to stop optimization
        )

        # Save results if logger_level mode is "INFO"
        if log_config.logger_level == "INFO":
            self._save_run_results()

        # Return optimization results
        return (
            self._study.best_params,  # Best parameters found
            self._study.best_value,  # Best objective value
            sum(self.metrics.called_obj) / len(self.metrics.called_obj),  # Success rate of trials
            self.metrics.func_evals_total[-1],  # Total number of function evaluations
        )
