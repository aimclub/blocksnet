import geopandas as gpd
from loguru import logger
from pandera.typing import Series
from pandera import Field
from ...common.validation import DfSchema


class BlocksSchema(DfSchema):

    l: Series[float] = Field(ge=0, default=0)
    fsi: Series[float] = Field(ge=0, default=0)
    mxi: Series[float] = Field(ge=0, default=0)

    @classmethod
    def _preprocess(cls, gdf: gpd.GeoDataFrame):
        if not "l" in gdf.columns:
            logger.warning("Column l not found in columns. Calculating from fsi and gsi.")
            if not "fsi" in gdf.columns:
                raise ValueError("Column fsi not found in columns.")
            if not "gsi" in gdf.columns:
                raise ValueError("Column gsi not found in columns.")
            gdf["l"] = gdf["fsi"] / gdf["gsi"]
        return gdf
