from enum import Enum
from .indicator_meta import IndicatorMeta


class IndicatorEnum(Enum):
    """Base enum for socio-economic indicators exposing metadata."""

    @property
    def meta(self) -> IndicatorMeta:
        """Return the indicator metadata descriptor."""
        return self.value

    def __repr__(self):
        repr = self.meta.name.capitalize().replace("_", " ")
        if self.meta.unit is not None:
            repr = f"{repr} ({self.meta.unit})"
        return repr
