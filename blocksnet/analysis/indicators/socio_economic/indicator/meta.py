from enum import Enum
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class IndicatorMeta:
    """IndicatorMeta class.

    """
    name: str
    per: Optional[Literal["capita", "area"]] = None
    unit: Optional[str] = None
    aggregatable: bool = True
