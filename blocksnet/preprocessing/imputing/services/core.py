import geopandas as gpd
import pandas as pd
from blocksnet.config import service_types_config
from .schemas import ServicesSchema


def _get_service_type_units(service_type: str) -> pd.DataFrame:
    if not service_type in service_types_config:
        raise ValueError(f"{service_type} not found in service types config")
    units = service_types_config.units
    return units[units.service_type == service_type].copy()


def impute_services(services_gdf: gpd.GeoDataFrame, service_type: str) -> gpd.GeoDataFrame:
    """Fill missing service capacity values based on configuration defaults.

    Parameters
    ----------
    services_gdf : geopandas.GeoDataFrame
        GeoDataFrame describing service locations.
    service_type : str
        Identifier of the service type whose default capacity should be used.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with missing ``capacity`` values imputed.

    Raises
    ------
    ValueError
        If the requested service type is not present in the configuration.
    """

    services_gdf = ServicesSchema(services_gdf)
    units = _get_service_type_units(service_type)
    impute_capacity = int(units.capacity.min())

    impute_mask = services_gdf.capacity.isna()
    impute_ids = impute_mask[impute_mask].index
    services_gdf.loc[impute_ids, "capacity"] = impute_capacity

    return services_gdf
