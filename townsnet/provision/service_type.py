import pandera as pa
import pandas as pd
import geopandas as gpd
import numpy as np
from enum import Enum
from pandera.typing import Series, Index
from pydantic import BaseModel, Field, field_validator

DB_ACCESSIBILITY_METERS_COLUMN = 'radius_availability_meters'
DB_ACCESSIBILITY_MINUTES_COLUMN = 'time_availability_minutes'
DB_SUPPLY_COUNT_COLUMN = 'services_per_1000_normative'
DB_SUPPLY_CAPACITY_COLUMN = 'services_capacity_per_1000_normative'

class AccessibilityType(Enum):
  METERS = 'м'
  MINUTES = 'мин'

class SupplyType(Enum):
  SERVICES_PER_1000 = 'шт. на 1000 человек'
  CAPACITY_PER_1000 = 'ед. на 1000 человек'

class Category(Enum):
  BASIC = 'Базовая'
  ADDITIONAL = 'Дополнительная'
  COMFORT = "Комфорт"

class ServiceTypesSchema(pa.DataFrameModel):
  idx : Index[int] = pa.Field(unique=True)
  name : Series[str]
  category : Series[str] = pa.Field(nullable=True)
  weight : Series[float] = pa.Field(nullable=True, coerce=True)
  
  @pa.parser('category')
  @classmethod
  def parse_category(cls, series : pd.Series) -> pd.Series:
      return series.apply(lambda s : s if s is None else str.upper)

class NormativesSchema(pa.DataFrameModel):
  service_type_id : Series[int]
  radius_availability_meters : Series[float] = pa.Field(coerce=True, nullable=True)
  time_availability_minutes : Series[float] = pa.Field(coerce=True, nullable=True)
  services_per_1000_normative : Series[float] = pa.Field(coerce=True, nullable=True)
  services_capacity_per_1000_normative : Series[float] = pa.Field(coerce=True, nullable=True)
  
  @pa.dataframe_check()
  @classmethod
  def check_availability(cls, df : pd.DataFrame) -> pd.Series:
      return ~df[DB_ACCESSIBILITY_METERS_COLUMN].isna() ^ ~df[DB_ACCESSIBILITY_MINUTES_COLUMN].isna()
  
  @pa.dataframe_check()
  @classmethod
  def check_supply(cls, df : pd.DataFrame) -> pd.Series:
      return ~df[DB_SUPPLY_COUNT_COLUMN].isna() ^ ~df[DB_SUPPLY_CAPACITY_COLUMN].isna()

class ServiceType(BaseModel):
  id : int
  accessibility_value : float = Field(ge=0)
  supply_value : float = Field(ge=0)
  accessibility_type : AccessibilityType
  supply_type : SupplyType
  category : Category | None
  weight : float = Field(ge=0, le=1)
  
  @field_validator('category', mode='before')
  @classmethod
  def validate_category(cls, c):
    if isinstance(c, str):
      return Category[c.upper()]
    return c

  @classmethod
  def from_series(cls, series : pd.Series):
    i = series['service_type_id']
    
    if not np.isnan(series[DB_ACCESSIBILITY_METERS_COLUMN]):
      accessibility_type = AccessibilityType.METERS
      accessibility_value = series[DB_ACCESSIBILITY_METERS_COLUMN]
    else:
      accessibility_type = AccessibilityType.MINUTES
      accessibility_value = series[DB_ACCESSIBILITY_MINUTES_COLUMN]

    if not np.isnan(series[DB_SUPPLY_COUNT_COLUMN]):
      supply_type = SupplyType.SERVICES_PER_1000
      supply_value = series[DB_SUPPLY_COUNT_COLUMN]
    else:
      supply_type = SupplyType.CAPACITY_PER_1000
      supply_value = series[DB_SUPPLY_CAPACITY_COLUMN]

    return cls(
      id = i,
      accessibility_value = accessibility_value,
      accessibility_type = accessibility_type, 
      supply_value = supply_value,
      supply_type = supply_type,
      weight = series['weight'],
      category=series['category']
    )

  @classmethod
  def initialize_service_types(cls, service_types : gpd.GeoDataFrame, normatives : gpd.GeoDataFrame) -> list :
    service_types = ServiceTypesSchema(service_types)
    normatives = NormativesSchema(normatives)
    service_types = service_types.merge(normatives, left_index=True, right_on='service_type_id')
    return [cls.from_series(s) for _, s in service_types.iterrows()]

  def __hash__(self):
    return self.id
