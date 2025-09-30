from enum import Enum
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class IndicatorMeta:
    """Metadata describing indicator naming, units, and aggregation rules."""
    name: str
    per: Optional[Literal["capita", "area"]] = None
    unit: Optional[str] = None
    aggregatable: bool = True
