import geopandas as gpd


def get_buildings(engine, city_crs, city_id):
    """
    TODO: add docstring
    """
    df_buildings = gpd.read_postgis(
        f"select population_balanced, building_area, living_area, storeys_count, is_living, "
        f"ST_Centroid(ST_Transform(geometry, {city_crs})) as geom from all_buildings where city_id={city_id}",
        con=engine,
    )
    df_buildings.rename(columns={"geom": "geometry"}, inplace=True)
    return df_buildings


def get_service(service_type, city_crs, engine, city_id):
    """
    TODO: add docstring
    """
    service_blocks_df = gpd.read_postgis(
        f"select capacity, geometry as geom from all_services where city_service_type_code in ('{service_type}') "
        f"and city_id={city_id}",
        con=engine,
    )

    service_blocks_df.rename(columns={"geom": "geometry"}, inplace=True)
    service_blocks_df["geometry"] = service_blocks_df["geometry"].convex_hull
    service_blocks_df["geometry"] = service_blocks_df["geometry"].to_crs(city_crs)
    service_blocks_df["geometry"] = service_blocks_df["geometry"].centroid
    service_blocks_df["geometry"] = service_blocks_df["geometry"].set_crs(city_crs)
    service_blocks_df = service_blocks_df.set_geometry("geometry")

    return service_blocks_df
