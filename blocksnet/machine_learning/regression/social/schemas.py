import pandas as pd
import shapely
from loguru import logger
from pandera import Field
from pandera.typing import Series

from ....utils.validation import DfSchema


class TechnicalIndicatorsSchema(DfSchema):
    longitude: Series[float]
    latitude: Series[float]
    population: Series[float] = Field(ge=0)
    site_area: Series[float] = Field(ge=0)
    # buidlings parameters
    is_living: Series[bool]
    footprint_area: Series[float] = Field(ge=0)
    build_floor_area: Series[float] = Field(ge=0)
    living_area: Series[float] = Field(ge=0)
    non_living_area: Series[float] = Field(ge=0)
    buildings_count: Series[float] = Field(ge=0)
    # land use areas
    agriculture: Series[float] = Field(ge=0)
    industrial: Series[float] = Field(ge=0)
    recreation: Series[float] = Field(ge=0)
    residential: Series[float] = Field(ge=0)
    special: Series[float] = Field(ge=0)
    business: Series[float] = Field(ge=0)
    transport: Series[float] = Field(ge=0)


class SocialIndicatorsSchema(DfSchema):
    nursing_home_count: Series[float] = Field(ge=0)
    hotel_count: Series[float] = Field(ge=0)
    theatre_count: Series[float] = Field(ge=0)
    cinema_count: Series[float] = Field(ge=0)
    secondary_vocational_education_institutions_count: Series[float] = Field(ge=0)
    university_count: Series[float] = Field(ge=0)
    stadium_count: Series[float] = Field(ge=0)
    emergency_medical_service_stations_count: Series[float] = Field(ge=0)
    kindergarten_count: Series[float] = Field(ge=0)
    hostel_count: Series[float] = Field(ge=0)
    park_count: Series[float] = Field(ge=0)
    multifunctional_center_count: Series[float] = Field(ge=0)
    pharmacy_count: Series[float] = Field(ge=0)
    sports_halls_count: Series[float] = Field(ge=0)
    hospital_count: Series[float] = Field(ge=0)
    school_count: Series[float] = Field(ge=0)
    mall_count: Series[float] = Field(ge=0)
    polyclinic_count: Series[float] = Field(ge=0)
    post_count: Series[float] = Field(ge=0)
    swimming_pool_count: Series[float] = Field(ge=0)
    library_count: Series[float] = Field(ge=0)
    guest_house_count: Series[float] = Field(ge=0)
    fire_safety_facilities_count: Series[float] = Field(ge=0)
    restaurant_count: Series[float] = Field(ge=0)
    police_count: Series[float] = Field(ge=0)
    museum_count: Series[float] = Field(ge=0)
    bank_count: Series[float] = Field(ge=0)
    pitch_count: Series[float] = Field(ge=0)
