import pytest
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from masterplan_tools import CityModel
from masterplan_tools.method.provision import LpProvision
from masterplan_tools.method.genetic.genetic import Genetic


local_crs = 32636
HECTARE_IN_SQUARE_METERS = 10_000
example_data_path = "./tests/data/city_model"


@pytest.fixture
def aggregated_blocks():
    return gpd.read_parquet(os.path.join(example_data_path, "aggr_blocks.parquet"))


@pytest.fixture
def accessibility_matrix():
    return pd.read_pickle(os.path.join(example_data_path, "accessibility_matrix.pickle"))


@pytest.fixture
def gdf(aggregated_blocks):
    gdf = aggregated_blocks[
        [
            "block_id",
            "geometry",
            "landuse",
            "is_living",
            "area",
            "current_living_area",
            "current_industrial_area",
            "current_green_area",
        ]
    ]
    gdf["free_area"] = (
        gdf["area"] * 0.8 - gdf["current_green_area"] - gdf["current_industrial_area"] - gdf["current_living_area"]
    ) / HECTARE_IN_SQUARE_METERS
    return gdf


@pytest.fixture
def services():
    schools = gpd.read_parquet(os.path.join(example_data_path, "schools.parquet"))
    kindergartens = gpd.read_parquet(os.path.join(example_data_path, "kindergartens.parquet"))
    recreational_areas = gpd.read_parquet(
        os.path.join(example_data_path, "recreational_areas.parquet")
    ).rename_geometry("geometry")

    hospitals = gpd.read_file(os.path.join(example_data_path, "hospitals.geojson"))
    pharmacies = gpd.read_file(os.path.join(example_data_path, "pharmacies.geojson"))
    policlinics = gpd.read_file(os.path.join(example_data_path, "policlinics.geojson"))

    services = {
        "schools": schools,
        "kindergartens": kindergartens,
        "recreational_areas": recreational_areas,
        "hospitals": hospitals,
        "pharmacies": pharmacies,
        "policlinics": policlinics,
    }
    return services


@pytest.fixture
def city_model(aggregated_blocks, accessibility_matrix, services):
    city_model = CityModel(blocks=aggregated_blocks, accessibility_matrix=accessibility_matrix, services=services)
    return city_model


@pytest.fixture
def all_services():
    all_services = {
        "schools": {250: 1.2, 300: 1.1, 600: 1.3, 800: 1.5, 1100: 1.8},
        "kindergartens": {180: 0.72, 250: 1.44, 280: 1.1},
        "recreational_areas": {100: 0.1, 500: 0.5, 1000: 1.0, 3000: 3.0, 5000: 5.0, 10000: 10.0, 15000: 15.0},
        "pharmacies": {1000: 0.005, 500: 0.0025, 1000: 0.0050, 1500: 0.0075},
        "hospitals": {60000: 1.5, 180000: 4.5, 272000: 6.8, 360000: 9, 600000: 15},
        "policlinics": {9615: 0.3, 19230: 0.6, 28846: 0.8, 32692: 0.9},
    }
    return all_services


@pytest.fixture
def scenario(scenario_list=["hospitals", "policlinics", "recreational_areas", "pharmacies"]):
    weights = None
    if not weights:
        weights = [round(1 / len(scenario_list), 2) for i in range(len(scenario_list))]

    scenario = dict(zip(scenario_list, weights))
    return scenario


@pytest.fixture
def lpp(city_model):
    lpp = LpProvision(city_model=city_model)
    return lpp


@pytest.fixture
def genetic(city_model, gdf, all_services, scenario):
    genetic = Genetic(city_model, gdf, all_services, scenario)
    return genetic


@pytest.fixture
def ga_params():
    ga_params = {
        "num_generations": 1,
        "num_parents_mating": 1,
        "sol_per_pop": 1,
        "parallel_processing": 12,
        "keep_parents": 1,
        "parent_selection_type": "tournament",
        "crossover_type": "scattered",
        "mutation_type": "adaptive",
        "mutation_percent_genes": (90, 10),
        "K_tournament": 1,
        "stop_criteria": "saturate_50",
    }
    return ga_params


@pytest.fixture
def solution(genetic, ga_params):
    _, updated_blocks = genetic.calculate_blocks_building_optinons(ga_params)
    return updated_blocks


@pytest.fixture
def genetic30(city_model, gdf, all_services, scenario):
    gdf_ = gdf[(gdf["landuse"] != "no_dev_area") & (gdf["free_area"] > 0.5)].sample(30)
    genetic30 = Genetic(city_model, gdf_, all_services, scenario)
    return genetic30


@pytest.fixture
def ga_params30():
    ga_params30 = {
        "num_generations": 1,
        "num_parents_mating": 1,
        "sol_per_pop": 1,
        "parallel_processing": 12,
        "keep_parents": 1,
        "parent_selection_type": "tournament",
        "crossover_type": "scattered",
        "mutation_type": "adaptive",
        "mutation_percent_genes": (90, 10),
        "K_tournament": 1,
        "stop_criteria": "saturate_50",
    }
    return ga_params30


@pytest.fixture
def solution30(genetic30, ga_params30):
    _, updated_blocks = genetic30.calculate_blocks_building_optinons(ga_params30)
    return updated_blocks


def test_updated_blocks(genetic, solution):
    assert list(solution.keys()) == genetic.BLOCKS["block_id"].tolist()


def test_metric(lpp, solution, scenario):
    _, mean1 = lpp.get_scenario_provisions(scenario)
    _, mean2 = lpp.get_scenario_provisions(scenario, solution)
    assert mean1 < mean2


def test_updated_blocks30(genetic30, solution30):
    assert list(solution30.keys()) == genetic30.BLOCKS["block_id"].tolist()


def test_metric30(lpp, solution30, scenario):
    _, mean1 = lpp.get_scenario_provisions(scenario)
    _, mean2 = lpp.get_scenario_provisions(scenario, solution30)
    assert mean1 < mean2
