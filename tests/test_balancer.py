import pytest
from masterplan_tools.method.balancing import MasterPlan


@pytest.fixture
def test_block():
    test_block = {
        "block_id": 1,
        "area": 100,
        "current_living_area": 0,
        "current_industrial_area": 0,
        "current_population": 0,
        "current_green_area": 0,
        "floors": 3
    }

    return test_block

@pytest.fixture
def test_block_params(test_block):
    max_living_area = test_block['area']*0.56
    max_industrial_area = test_block['area']*0.24
    max_free_area = test_block['area']*0.2
    shoolkids_ratio = 0.12
    kids_ratio = 0.061
    shoolkids_requirement = 100
    kids_requirement = 140
    return {'max_living_area': max_living_area, 'max_industrial_area': max_industrial_area, 'max_free_area': max_free_area,
            'shoolkids_ratio': shoolkids_ratio, 'kids_ratio': kids_ratio, 
            'shoolkids_requirement': shoolkids_requirement, 'kids_requirement': kids_requirement}

@pytest.fixture
def test_solution(test_block):
    mp = MasterPlan(
        area=test_block["area"],
        current_living_area=test_block["current_living_area"],
        current_industrial_area=test_block["current_industrial_area"],
        current_population=test_block["current_population"],
        current_green_area=test_block["current_green_area"],
    )

    test_solution = mp.optimal_solution_indicators()
    return test_solution


def test_living_area_bounds(test_solution, test_block_params):
    assert (test_solution['parking1_area'] + test_solution['living_area']) < test_block_params['max_living_area']

def test_industrial_area_bounds(test_solution, test_block_params):
    assert (test_solution['parking2_area'] + test_solution['schools_area'] + test_solution['kindergartens_area']) < test_block_params['max_industrial_area']

def test_free_area_bounds(test_solution, test_block_params):
    assert (test_solution['op_area'] + test_solution['green_area']) < test_block_params['max_free_area']

def test_kindergartens_capacity(test_solution, test_block_params):
    assert (test_solution['population']*test_block_params['kids_ratio'] - test_solution['kindergartens_capacity']) < test_block_params['kids_requirement']

def test_schools_capacity(test_solution, test_block_params):
    assert (test_solution['population']*test_block_params['shoolkids_ratio'] - test_solution['schools_capacity']) < test_block_params['shoolkids_requirement']