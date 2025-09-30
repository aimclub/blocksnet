from enum import unique
from ..indicator_enum import IndicatorEnum
from ..indicator_meta import IndicatorMeta


@unique
class TransportIndicator(IndicatorEnum):
    """Transport accessibility and infrastructure indicators."""
    # road network
    ROAD_NETWORK_DENSITY = IndicatorMeta("road_network_density", per="area", unit="km/km2")
    SETTLEMENTS_CONNECTIVITY = IndicatorMeta("settlements_connectivity", aggregatable=False)
    ROAD_NETWORK_LENGTH = IndicatorMeta("road_network_length", unit="km")

    # distance
    AVERAGE_DISTANCE_TO_REGIONAL_CENTER = IndicatorMeta("average_distance_to_regional_center", aggregatable=False)
    AVERAGE_DISTANCE_TO_FEDERAL_HIGHWAYS = IndicatorMeta("average_distance_to_federal_highways", aggregatable=False)

    # fuel stations
    FUEL_STATIONS_COUNT = IndicatorMeta("fuel_stations_count")
    AVERAGE_FUEL_STATION_ACCESSIBILITY = IndicatorMeta(
        "average_fuel_station_accessibility", aggregatable=False, unit="h"
    )

    # railway
    RAILWAY_STOPS_COUNT = IndicatorMeta("railway_stops_count")
    AVERAGE_RAILWAY_STOP_ACCESSIBILITY = IndicatorMeta(
        "average_railway_stop_accessibility", aggregatable=False, unit="h"
    )

    # airports
    INTERNATIONAL_AIRPORTS_COUNT = IndicatorMeta("international_airports_count")
    AVERAGE_INTERNATIONAL_AIRPORT_ACCESSIBILITY = IndicatorMeta(
        "average_international_airport_accessibility", aggregatable=False
    )

    REGIONAL_AIRPORTS_COUNT = IndicatorMeta("regional_airports_count")
    AVERAGE_REGIONAL_AIRPORT_ACCESSIBILITY = IndicatorMeta("average_regional_airport_accessibility", aggregatable=False)

    # ports
    PORTS_COUNT = IndicatorMeta("ports_count")
