"""Testing blocks cutter behavior"""

import os
import pytest
import geopandas as gpd
import pandas as pd
import statistics
from blocksnet import City, Provision, Accessibility, Connectivity, Genetic

data_path = "./tests/data/city_model"
local_crs = 32636


@pytest.fixture
def blocks():
    return gpd.read_parquet(os.path.join(data_path, "aggr_blocks.parquet")).to_crs(local_crs)


@pytest.fixture
def adjacency_matrix():
    return pd.read_pickle(os.path.join(data_path, "accessibility_matrix.pickle"))


@pytest.fixture
def services():
    schools = gpd.read_parquet(os.path.join(data_path, "schools.parquet")).to_crs(local_crs)
    kindergartens = gpd.read_parquet(os.path.join(data_path, "kindergartens.parquet")).to_crs(local_crs)
    return {"school": schools, "kindergarten": kindergartens}


@pytest.fixture
def buildings():
    buildings = gpd.read_parquet(os.path.join(data_path, "buildings.parquet")).to_crs(local_crs)
    return buildings.rename(
        columns={"storeys_count": "floors", "population_balanced": "population", "total_area": "area"}
    )


@pytest.fixture
def scenario():
    return {"school": 0.6, "kindergarten": 0.4}


@pytest.fixture
def selected_blocks():
    return [82, 83, 84]


@pytest.fixture
def update_services():
    update = {218: {"kindergarten": 1000}, 30: {"school": 1000}}
    return pd.DataFrame.from_dict(update, orient="index")


@pytest.fixture
def update_population():
    update = {30: {"population": 1000}}
    return pd.DataFrame.from_dict(update, orient="index")


@pytest.fixture
def city_model(blocks, adjacency_matrix, services, buildings):
    city = City(blocks, adjacency_matrix)
    city.update_buildings(buildings)
    for service_type, gdf in services.items():
        city.update_services(service_type, gdf)
    return city


def test_coherence(city_model, blocks):
    n = len(city_model.adjacency_matrix.index)
    m = len(city_model.adjacency_matrix.columns)
    assert len(city_model.blocks) == n == m
    assert len(city_model.blocks) == len(blocks.index)


def test_provision_lp(city_model, update_services, update_population):
    provision = Provision(city_model=city_model)
    calc, _ = provision.calculate("school")
    calc_services, _ = provision.calculate("school", update_services)
    calc_population, _ = provision.calculate("school", update_population)
    total = provision.total(calc)
    total_services = provision.total(calc_services)
    total_population = provision.total(calc_population)
    assert total <= total_services
    assert total >= total_population


def test_provision_iterative(city_model, update_services, update_population):
    provision = Provision(city_model=city_model)
    calc, _ = provision.calculate("kindergarten", method="iterative")
    calc_services, _ = provision.calculate("kindergarten", update_services, method="iterative")
    calc_population, _ = provision.calculate("kindergarten", update_population, method="iterative")
    total = provision.total(calc)
    total_services = provision.total(calc_services)
    total_population = provision.total(calc_population)
    assert total <= total_services
    assert total >= total_population


def test_genetic(city_model, scenario, selected_blocks):
    provision = Provision(city_model=city_model)
    genetic = Genetic(city_model=city_model, SCENARIO=scenario)
    _, total_before = provision.calculate_scenario(scenario)
    res = genetic.calculate(3, selected_blocks=selected_blocks)
    update_df = pd.DataFrame.from_dict(res, orient="index")
    _, total_after = provision.calculate_scenario(scenario, update_df)
    assert total_after >= total_before


def test_spatial(city_model):
    accessibility = Accessibility(city_model=city_model)
    connectivity = Connectivity(city_model=city_model)
    block = city_model.blocks[0]
    acc_calc = accessibility.calculate(block)
    conn_calc = connectivity.calculate()
    assert statistics.median(acc_calc.loc[acc_calc.index != block.id]["distance"]) == conn_calc.loc[block.id]["median"]
