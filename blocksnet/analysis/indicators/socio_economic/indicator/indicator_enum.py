from enum import Enum
from typing import Any
from .indicator_meta import IndicatorMeta


class IndicatorEnum(Enum):
    @property
    def meta(self) -> IndicatorMeta:
        return self.value
