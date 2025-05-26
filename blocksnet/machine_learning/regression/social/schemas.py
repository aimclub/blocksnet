import shapely
import pandas as pd
from pandera import Field
from pandera.typing import Series
from loguru import logger
from ....utils.validation import DfSchema

class TechnicalIndicatorsSchema(DfSchema):
    longitude: Series[float]
    latitude: Series[float]
    population: Series[float] = Field(ge=0)
    area: Series[float] = Field(ge=0)
    is_living: Series[bool]
    footprint_area: Series[float] = Field(ge=0)
    build_floor_area: Series[float] = Field(ge=0)
    living_area: Series[float] = Field(ge=0)
    non_living_area: Series[float] = Field(ge=0)
    buildings_count: Series[float] = Field(ge=0)
    agriculture: Series[float] = Field(ge=0)
    industrial: Series[float] = Field(ge=0)
    recreation: Series[float] = Field(ge=0)
    residential: Series[float] = Field(ge=0)
    special: Series[float] = Field(ge=0)
    business: Series[float] = Field(ge=0)
    transport: Series[float] = Field(ge=0)

class SocialIndicatorsScheme(DfSchema):
    number_of_nursing_homes: Series[float] = Field(ge=0)
    number_of_hotels: Series[float] = Field(ge=0)
    number_of_theaters: Series[float] = Field(ge=0)
    number_of_cinemas: Series[float] = Field(ge=0)
    number_of_secondary_vocational_education_institutions: Series[float] = Field(ge=0)
    number_of_higher_education_institutions: Series[float] = Field(ge=0)
    number_of_stadiums: Series[float] = Field(ge=0)
    number_of_emergency_medical_service_stations: Series[float] = Field(ge=0)
    number_of_preschool_educational_institutions: Series[float] = Field(ge=0)
    number_of_hostels: Series[float] = Field(ge=0)
    number_of_parks: Series[float] = Field(ge=0)
    number_of_public_and_municipal_service_centers: Series[float] = Field(ge=0)
    number_of_pharmacies: Series[float] = Field(ge=0)
    number_of_sports_halls: Series[float] = Field(ge=0)
    number_of_hospitals: Series[float] = Field(ge=0)
    number_of_general_education_institutions: Series[float] = Field(ge=0)
    number_of_shopping_and_entertainment_centers: Series[float] = Field(ge=0)
    number_of_outpatient_clinics: Series[float] = Field(ge=0)
    number_of_post_offices: Series[float] = Field(ge=0)
    number_of_swimming_pools: Series[float] = Field(ge=0)
    number_of_libraries: Series[float] = Field(ge=0)
    number_of_tourist_bases: Series[float] = Field(ge=0)
    number_of_fire_safety_facilities: Series[float] = Field(ge=0)
    number_of_catering_facilities: Series[float] = Field(ge=0)
    number_of_police_stations: Series[float] = Field(ge=0)
    number_of_museums: Series[float] = Field(ge=0)
    number_of_bank_branches: Series[float] = Field(ge=0)
    number_of_flat_sports_facilities: Series[float] = Field(ge=0)