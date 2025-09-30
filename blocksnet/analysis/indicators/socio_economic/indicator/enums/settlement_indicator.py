from enum import unique
from ..enum import IndicatorEnum
from ..meta import IndicatorMeta


@unique
class SettlementIndicator(IndicatorEnum):
    # population
    """SettlementIndicator class.

    """
    URBAN_POPULATION = IndicatorMeta("urban_population")
    RURAL_POPULATION = IndicatorMeta("rural_population")

    # rural settlements
    LARGE_RURAL_SETTLEMENTS = IndicatorMeta("large_rural_settlements")  # >3k population
    BIG_RURAL_SETTLEMENTS = IndicatorMeta("big_rural_settlements")  # 1-3k
    MEDIUM_RURAL_SETTLEMENTS = IndicatorMeta("medium_rural_settlements")  # 200-1k
    SMALL_RURAL_SETTLEMENTS = IndicatorMeta("small_rural_settlements")  # <200

    # cities by size
    SUPER_LARGE_CITIES = IndicatorMeta("super_large_cities")  # >3M population
    LARGEST_CITIES = IndicatorMeta("largest_cities")  # 1–3M
    LARGE_CITIES = IndicatorMeta("large_cities")  # 250k–1M
    BIG_CITIES = IndicatorMeta("big_cities")  # 100–250k
    MEDIUM_CITIES = IndicatorMeta("medium_cities")  # 50–100k
    SMALL_CITIES = IndicatorMeta("small_cities")  # <50k

    # special settlements
    SPECIAL_SETTLEMENTS = IndicatorMeta("special_settlements")

    # agglomerations
    SETTLEMENTS_IN_AGGLOMERATIONS = IndicatorMeta("settlements_in_agglomerations")
    SETTLEMENTS_OUTSIDE_AGGLOMERATIONS = IndicatorMeta("settlements_outside_agglomerations")
