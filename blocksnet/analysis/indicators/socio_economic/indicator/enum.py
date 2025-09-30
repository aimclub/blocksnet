from enum import Enum
from typing import Any
from .meta import IndicatorMeta


class IndicatorEnum(Enum):
    @property
    """IndicatorEnum class.

    """
    def meta(self) -> IndicatorMeta:
        """Meta.

        Returns
        -------
        IndicatorMeta
            Description.

        """
        return self.value
