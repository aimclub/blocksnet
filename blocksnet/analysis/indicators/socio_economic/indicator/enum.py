from enum import Enum
from typing import Any
from .meta import IndicatorMeta


class IndicatorEnum(Enum):
    @property
    def meta(self) -> IndicatorMeta:
        return self.value
