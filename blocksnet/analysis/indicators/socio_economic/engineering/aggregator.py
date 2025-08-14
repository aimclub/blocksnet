import pandas as pd
from .indicator import EngineeringIndicator
from ..base_aggregator import BaseAggregator


class EngineeringAggregator(BaseAggregator):

    indicator_cls = EngineeringIndicator
