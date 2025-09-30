from enum import Enum


class SettlementCategory(Enum):
    """Administrative settlement scales supported by BlocksNet."""

    TOWN = "town"
    SUBURB = "suburb"
    VILLAGE = "village"
