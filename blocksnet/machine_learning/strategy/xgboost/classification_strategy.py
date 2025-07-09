from .base_strategy import XGBoostBaseStrategy


class XGBoostClassificationStrategy(XGBoostBaseStrategy):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TODO: to be implemented in the future")

    def _get_eval_metric(self) -> str:
        return super()._get_eval_metric("error")
