from catboost import CatBoostRegressor
from .base_strategy import CatBoostBaseStrategy


class CatBoostRegressionStrategy(CatBoostBaseStrategy):
    """CatBoostRegressionStrategy class.

    """
    def __init__(self, model_params: dict | None = None):
        """Initialize the instance.

        Parameters
        ----------
        model_params : dict | None, default: None
            Description.

        Returns
        -------
        None
            Description.

        """
        super().__init__(CatBoostRegressor, model_params)
