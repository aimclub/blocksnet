import pytest
from shapely import Polygon, Point, distance
import geopandas as gpd
import pandas as pd
from townsnet import Region, Provision

crs = 32636

grid_district_size = 50
grid_settlement_size = 10
grid_min = 0
grid_max = 100

default_capacity = 100

@pytest.fixture
def districts():
    polygons = []
    for x in range(grid_min, grid_max, grid_district_size):
        for y in range(grid_min, grid_max, grid_district_size):
            polygon = Polygon.from_bounds(x,y,x+grid_district_size, y+grid_district_size)
            polygons.append(polygon)
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf['name'] = gdf.apply(lambda s : f"district_{s.name}", axis=1)
    return gdf

@pytest.fixture
def settlements():
    polygons = []
    for x in range(grid_min, grid_max, grid_settlement_size):
        for y in range(grid_min, grid_max, grid_settlement_size):
            polygon = Polygon.from_bounds(x,y,x+grid_settlement_size, y+grid_settlement_size)
            polygons.append(polygon)
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf['name'] = gdf.apply(lambda s : f"settlement_{s.name}", axis=1)
    return gdf

@pytest.fixture
def towns(settlements):
    gdf = settlements.copy()
    gdf.geometry = gdf.geometry.representative_point()
    gdf['name'] = gdf.apply(lambda s : f"town_{s.name}", axis=1)
    gdf['population'] = gdf.apply(lambda s : (s.name+1)*1000, axis=1)
    return gdf

@pytest.fixture
def territories():
    polygons = []
    for x in range(grid_min, grid_max, grid_district_size):
        for y in range(grid_min, grid_max, grid_district_size):
            polygon = Polygon.from_bounds(x+grid_settlement_size/4,y+grid_settlement_size/4,x+grid_settlement_size/2, y+grid_settlement_size/2)
            polygons.append(polygon)
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf['name'] = gdf.apply(lambda s : f"territory_{s.name}", axis=1)
    return gdf

@pytest.fixture
def accessibility_matrix(towns):
    acc_mx = pd.DataFrame(0.0, index=towns.index, columns=towns.index)
    for i in acc_mx.index:
        for j in acc_mx.columns:
            acc_mx.loc[i,j] = distance(towns.loc[i, 'geometry'], towns.loc[j, 'geometry'])/4
    return acc_mx

@pytest.fixture
def region(districts, settlements, towns, accessibility_matrix, territories):
    region = Region(districts=districts, settlements=settlements, towns=towns, accessibility_matrix=accessibility_matrix, territories=territories)
    for service_type in region.service_types:
        gdf = region.get_towns_gdf()[['geometry']]
        gdf['capacity'] = default_capacity
        gdf = region.match_services_towns(gdf)
        region.update_services(service_type, gdf)
    return region

@pytest.fixture
def provisions(region):
    prov = Provision(region=region)
    provs = {}
    for st in region.service_types:
        provs[st] = prov.calculate(st)
    return provs

def test_provisions(provisions, region):
    for prov in provisions.values():
        districts_gdf, settlements_gdf, towns_gdf, links_gdf = prov
        assert all(districts_gdf.index == region.districts.index)
        assert all(settlements_gdf.index == region.settlements.index)
        assert all(towns_gdf.index == region.get_towns_gdf().index)
        for gdf in [districts_gdf, settlements_gdf, towns_gdf]:
            assert all(gdf.provision>=0) and all(gdf.provision<= 1)
            assert all(gdf.demand >= 0)
            assert all(gdf.demand_left >= 0)
            assert all(gdf.demand_within >= 0)
            assert all(gdf.demand_without >= 0)
            assert all(gdf.capacity >= 0)
            assert all(gdf.capacity_left >= 0)
        assert all(links_gdf.demand >= 0)
        assert all(links_gdf['from'].isin(region.get_towns_gdf().index))
        assert all(links_gdf['to'].isin(region.get_towns_gdf().index))

def test_context_provision(region, provisions):
    for territory in region.territories:
        _, value = territory.get_indicators(provisions)
        assert 0 <= value <= 11