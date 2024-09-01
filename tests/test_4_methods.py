"""Testing methods behavior"""

import os
import pytest
import geopandas as gpd
import pandas as pd
from blocksnet import City
from blocksnet.method.accessibility import Accessibility, ACCESSIBILITY_FROM_COLUMN, ACCESSIBILITY_TO_COLUMN
from blocksnet.method.connectivity import Connectivity, CONNECTIVITY_COLUMN

# from blocksnet.method.diversity import Diversity, DIVERSITY_COLUMN
from blocksnet.method.centrality import (
    Centrality,
    PopulationCentrality,
    CENTRALITY_COLUMN,
    POPULATION_CENTRALITY_COLUMN,
)
from blocksnet.method.spacematrix import Spacematrix, SM_CLUSTER_COLUMN, SM_MORPHOTYPE_COLUMN, KB_MORPHOTYPE_COLUMN
from blocksnet.methodland_use_prediction import LandUsePrediction, PREDICTION_COLUMN
from blocksnet.models.land_use import LandUse

data_path = "./tests/data"

# input data


@pytest.fixture
def city():
    return City.from_pickle(os.path.join(data_path, "_city.pickle"))


@pytest.fixture
def crs(city):
    return city.crs


@pytest.fixture
def block(city):
    return city[123]


@pytest.fixture
def service_type(city):
    return city["school"]


# accessibility


@pytest.fixture
def accessibility(city):
    return Accessibility(city_model=city)


@pytest.fixture
def accessibility_result(accessibility, block):
    return accessibility.calculate(block)


def test_accessibility(accessibility_result):
    assert all(accessibility_result[ACCESSIBILITY_FROM_COLUMN] >= 0) and all(
        accessibility_result[ACCESSIBILITY_TO_COLUMN] >= 0
    )


# connectivity


@pytest.fixture
def connectivity(city):
    return Connectivity(city_model=city)


@pytest.fixture
def connectivity_result(connectivity):
    return connectivity.calculate()


def test_connectivity(connectivity_result):
    assert all(connectivity_result[CONNECTIVITY_COLUMN] >= 0)


# diversity

# @pytest.fixture
# def diversity(city):
#     return Diversity(city_model=city)


# @pytest.fixture
# def diversity_result(diversity):
#     return diversity.calculate()

# def test_diversity(diversity_result):
#     assert all(diversity_result[DIVERSITY_COLUMN] >= 0)

# centrality

# @pytest.fixture
# def centrality(city):
#     return Centrality(city_model=city)

# @pytest.fixture
# def centrality_result(centrality):
#     return centrality.calculate()

# def test_centrality(centrality_result):
#     assert all(centrality_result[CENTRALITY_COLUMN] >= 0) and all(centrality_result[CENTRALITY_COLUMN] <= 3)

# population centrality


@pytest.fixture
def population_centrality(city):
    return PopulationCentrality(city_model=city)


@pytest.fixture
def population_centrality_result(population_centrality):
    return population_centrality.calculate()


def test_population_centrality(population_centrality_result):
    res = population_centrality_result
    assert all(res[~res[POPULATION_CENTRALITY_COLUMN].isna()][POPULATION_CENTRALITY_COLUMN] >= 0)
    assert all(res[~res[POPULATION_CENTRALITY_COLUMN].isna()][POPULATION_CENTRALITY_COLUMN] <= 10)


# spacematrix


@pytest.fixture
def spacematrix(city):
    return Spacematrix(city_model=city)


@pytest.fixture
def spacematrix_result(spacematrix):
    return spacematrix.calculate()


def test_spacematrix(spacematrix_result):
    res = spacematrix_result
    assert "fsi" in res.columns
    assert "mxi" in res.columns
    assert "l" in res.columns


# land use prediction


@pytest.fixture
def lup(city):
    return LandUsePrediction(city_model=city)


@pytest.fixture
def lup_result(lup):
    return lup.calculate()


def test_lup(lup_result):
    res = lup_result
    lus = [lu.name for lu in LandUse]
    assert all(res[~res[PREDICTION_COLUMN].isna()].isin(lus))
