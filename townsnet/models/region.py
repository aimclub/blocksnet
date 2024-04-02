from pydantic import BaseModel, InstanceOf, model_validator, field_validator
import shapely
from .geodataframe import GeoDataFrame, BaseRow

class RayonRow(BaseRow):
    geometry : shapely.Polygon | shapely.MultiPolygon

class OkrugRow(BaseRow):
    geometry : shapely.Polygon | shapely.MultiPolygon
    population : int

class TownRow(BaseRow):
    geometry : shapely.Point

class Town(BaseModel):
    geometry : InstanceOf[shapely.Point]
    population : int

    @classmethod
    def from_gdf(cls, gdf):
        return [cls(**gdf.loc[i].to_dict()) for i in gdf.index]

class Region(BaseModel):
    rayons : GeoDataFrame[RayonRow]
    okrugs : GeoDataFrame[OkrugRow]
    towns : list[Town]

    @field_validator('rayons', mode='before')
    @classmethod
    def validate_rayons(gdf):
        if not isinstance(gdf, GeoDataFrame[RayonRow]):
            gdf = GeoDataFrame[RayonRow](gdf)
        return gdf
    
    @field_validator('okrugs', mode='before')
    @classmethod
    def validate_okrugs(gdf):
        if not isinstance(gdf, GeoDataFrame[OkrugRow]):
            gdf = GeoDataFrame[OkrugRow](gdf)
        return gdf

    @field_validator('towns', mode='before')
    @classmethod
    def validate_towns(gdf):
        if not isinstance(gdf, GeoDataFrame[TownRow]):
            gdf = GeoDataFrame[OkrugRow](gdf)
        return gdf

    @model_validator(mode='before')
    @classmethod
    def distribute_population(cls, data : any) -> any:
        ...
    
        
