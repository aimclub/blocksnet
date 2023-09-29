"""Testing blocks cutter behavior"""

import os
import pytest
import geopandas as gpd
import pandas as pd
from blocksnet.models import CityModel
from blocksnet.method.provision import LpProvision, ProvisionModel

data_path = "./tests/data/city_model"
local_crs = 32636


@pytest.fixture
def aggr_blocks():
    return gpd.read_parquet(os.path.join(data_path, "aggr_blocks.parquet")).to_crs(local_crs)


@pytest.fixture
def accessibility_matrix():
    return pd.read_pickle(os.path.join(data_path, "accessibility_matrix.pickle"))


@pytest.fixture
def services():
    schools = gpd.read_parquet(os.path.join(data_path, "schools.parquet")).to_crs(local_crs)
    kindergartens = gpd.read_parquet(os.path.join(data_path, "kindergartens.parquet")).to_crs(local_crs)
    return {"schools": schools, "kindergartens": kindergartens}


@pytest.fixture
def scenario():
    return {
        "schools": 0.25,
        "kindergartens": 0.75,
    }


@pytest.fixture
def updated_blocks_services():
    return {218: {"kindergartens": 1000}, 30: {"schools": 1000}}


@pytest.fixture
def updated_blocks_population():
    return {30: {"population": 1000}}


@pytest.fixture
def city_model(aggr_blocks, accessibility_matrix, services):
    return CityModel(accessibility_matrix=accessibility_matrix, blocks=aggr_blocks, services=services)


def test_matrix(city_model):
    acc_mx = city_model.accessibility_matrix.df
    blocks = city_model.blocks.to_gdf()
    assert (acc_mx.index == blocks.index).all()
    assert (acc_mx.columns == blocks.index).all()
    assert (blocks.index == blocks["block_id"]).all()


def test_graph(city_model):
    graph = city_model.services_graph
    blocks = city_model.blocks.to_gdf()
    assert sorted(graph.nodes) == sorted(blocks.index)


def test_lp_provision(city_model, scenario, updated_blocks_services, updated_blocks_population):
    lpp = LpProvision(city_model=city_model)
    _, mean = lpp.get_scenario_provisions(scenario)
    _, servicesMean = lpp.get_scenario_provisions(scenario, updated_blocks_services)
    assert mean <= servicesMean
    _, populationMean = lpp.get_scenario_provisions(scenario, updated_blocks_population)
    assert mean >= populationMean


def test_iterative_provision(city_model):
    updated_block = {"block_id": 242, "population": 0, "is_kindergartens_service": 1, "kindergartens_capacity": 500}
    provision = ProvisionModel(city_model=city_model, service_name="kindergartens")
    prov = provision.run(overflow=True)
    prov_before = prov[f"demand_{'kindergartens'}"].sum() / prov[f"population_prov_{'kindergartens'}"].sum()
    initial_graph = city_model.services_graph
    updated_graph = initial_graph.copy()
    if updated_block["block_id"] in city_model.services_graph.nodes:
        for attr_name, attr_value in updated_block.items():
            if attr_name in updated_graph.nodes[updated_block["block_id"]]:
                updated_graph.nodes[updated_block["block_id"]][attr_name] += attr_value
    city_model.services_graph = updated_graph
    prov = provision.run(overflow=True)
    prov_after = prov[f"demand_{'kindergartens'}"].sum() / prov[f"population_prov_{'kindergartens'}"].sum()
    city_model.services_graph = initial_graph
    assert prov_before <= prov_after
