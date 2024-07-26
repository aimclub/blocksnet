"""Testing methods behavior"""

import os
import pytest
import geopandas as gpd
import pandas as pd
from blocksnet import City
from blocksnet.method.accessibility import Accessibility, ACCESSIBILITY_FROM_COLUMN, ACCESSIBILITY_TO_COLUMN
from blocksnet.method.connectivity import Connectivity, CONNECTIVITY_COLUMN

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
