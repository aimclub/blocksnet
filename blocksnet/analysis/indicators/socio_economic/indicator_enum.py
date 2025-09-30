from enum import Enum
from .indicator_meta import IndicatorMeta


class IndicatorEnum(Enum):
    @property
    def meta(self) -> IndicatorMeta:
        return self.value

    def __repr__(self):
        repr = self.meta.name.capitalize().replace("_", " ")
        if self.meta.unit is not None:
            repr = f"{repr} ({self.meta.unit})"
        return repr
