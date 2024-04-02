import geopandas as gpd
from pydantic import BaseModel, Field, field_validator, model_validator
from shapely import Point
from ..models.geodataframe import GeoDataFrame, BaseRow
from ..models.service_type import ServiceType
from ..utils.const import SQUARE_METERS_IN_HECTARE


class ServiceRow(BaseRow):
    geometry: Point
    area: float = Field(ge=0)
    """Service area in square meters"""
    area_ha: float = Field(ge=0)
    """Service area in hectares"""


class CapacityFiller(BaseModel):
    services: GeoDataFrame[ServiceRow]
    service_type: ServiceType
    is_osm: bool = False

    @staticmethod
    def _fill_area(series) -> float:
        """Return geometry area if service area is set to 0 or NaN"""
        area = series["area"]
        geometry = series["geometry"]
        if not area > 0:
            area = geometry.area
        return area

    @staticmethod
    def _union_geometries(gdf) -> gpd.GeoDataFrame:
        """
        Since OSM geometries may intersect, the proposed method unions overlayed
        geometries to one object
        """
        crs = gdf.crs
        services = (
            gpd.GeoDataFrame([{"geometry": gdf.geometry.unary_union}], crs=crs)
            .explode("geometry", index_parts=False)
            .reset_index(drop=True)
        )
        services["area"] = services.area
        return services

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, data):
        services = data["services"]
        is_osm = data["is_osm"]
        if is_osm:
            data["services"] = cls._union_geometries(services)
        else:
            data["services"] = services.copy()
        return data

    @field_validator("services", mode="before")
    @classmethod
    def validate_services(cls, gdf):
        gdf["area"] = gdf.apply(cls._fill_area, axis=1)
        gdf["area_ha"] = gdf["area"] / SQUARE_METERS_IN_HECTARE
        gdf.geometry = gdf.geometry.representative_point()
        return GeoDataFrame[ServiceRow](gdf)

    @property
    def _integrated_bricks(self):
        bricks = self.service_type.bricks
        filtered = filter(lambda x: x.is_integrated, bricks)
        return list(filtered)

    @property
    def _non_integrated_bricks(self):
        bricks = self.service_type.bricks
        filtered = filter(lambda x: not x.is_integrated, bricks)
        return list(filtered)

    @staticmethod
    def _get_similar_brick(area, bricks):
        return min(bricks, key=lambda x: abs(x.area - area))

    def _fill_capacity(self, area):
        bricks = []
        if area > 0:
            bricks = self._non_integrated_bricks
            if len(bricks) == 0:
                bricks = self._integrated_bricks
        else:
            bricks = self._integrated_bricks
            if len(bricks) == 0:
                bricks = self._non_integrated_bricks
        brick = self._get_similar_brick(area, bricks)
        return brick.capacity

    def fill(self) -> gpd.GeoDataFrame:
        services = self.services.copy()
        # get_capacity = lambda x : self._fill_capacity(x)
        services["capacity"] = services["area_ha"].apply(self._fill_capacity)
        return services
