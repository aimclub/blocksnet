from functools import wraps
import pandas as pd
from loguru import logger
from tqdm import tqdm
from pulp import PULP_CBC_CMD, LpMaximize, LpProblem, LpVariable, lpSum, LpInteger
from .schemas import BlocksSchema
from ....utils import validation

DEMAND_COLUMN = "demand"
DEMAND_LEFT_COLUMN = "demand_left"
DEMAND_WITHIN_COLUMN = "demand_within"
DEMAND_WITHOUT_COLUMN = "demand_without"

CAPACITY_LEFT_COLUMN = "capacity_left"
CAPACITY_WITHIN_COLUMN = "capacity_within"
CAPACITY_WITHOUT_COLUMN = "capacity_without"

PROVISION_COLUMN = "provision"


def _initialize_provision_df(blocks_df: pd.DataFrame, demand: int):
    logger.info("Initializing provision DataFrame")
    blocks_df[DEMAND_COLUMN] = blocks_df.population.apply(lambda p: round(p / 1000 * demand))
    blocks_df = blocks_df.assign(
        **{
            DEMAND_LEFT_COLUMN: blocks_df[DEMAND_COLUMN],
            DEMAND_WITHIN_COLUMN: 0,
            DEMAND_WITHOUT_COLUMN: 0,
            CAPACITY_LEFT_COLUMN: blocks_df.capacity,
            CAPACITY_WITHIN_COLUMN: 0,
            CAPACITY_WITHOUT_COLUMN: 0,
        }
    )
    return blocks_df


def _validate_and_preprocess_input(func):
    @wraps(func)
    def wrapper(blocks_df: pd.DataFrame, accessibility_matrix: pd.DataFrame, *args, **kwargs):
        validation.validate_accessibility_matrix(accessibility_matrix, blocks_df)
        blocks_df = BlocksSchema(blocks_df)
        return func(blocks_df, accessibility_matrix, *args, **kwargs)

    return wrapper


def _supply_self(blocks_df: pd.DataFrame):
    logger.info("Supplying blocks with own capacities")
    supply = blocks_df.apply(lambda s: min(s[DEMAND_COLUMN], s.capacity), axis=1)
    blocks_df[DEMAND_WITHIN_COLUMN] += supply
    blocks_df[DEMAND_LEFT_COLUMN] -= supply
    blocks_df[CAPACITY_LEFT_COLUMN] -= supply


def _get_distance(id1: int, id2: int, accessibility_matrix: pd.DataFrame):
    distance = accessibility_matrix.loc[id1, id2]
    return max(distance, 1)


import numpy as np


def _set_lp_problem(blocks_df: pd.DataFrame, accessibility_matrix: pd.DataFrame, selection_range: int):

    demand_blocks = blocks_df.loc[blocks_df[DEMAND_LEFT_COLUMN] > 0]
    capacity_blocks = blocks_df.loc[blocks_df[CAPACITY_LEFT_COLUMN] > 0]

    def _get_weight(id1: int, id2: int):
        distance = _get_distance(id1, id2, accessibility_matrix)
        return demand_blocks.loc[id1, DEMAND_LEFT_COLUMN] / (distance**2)

    prob = LpProblem("Provision", LpMaximize)
    products = [
        (i, j)
        for i in demand_blocks.index
        for j in capacity_blocks.index
        if _get_distance(i, j, accessibility_matrix) <= selection_range
    ]

    x = LpVariable.dicts("link", products, 0, None, cat=LpInteger)

    prob += lpSum(_get_weight(n, m) * x[n, m] for n, m in products)

    demand_constraints = {n: [] for n in demand_blocks.index}
    capacity_constraints = {m: [] for m in capacity_blocks.index}

    for n, m in products:
        demand_constraints[n].append(x[n, m])
        capacity_constraints[m].append(x[n, m])

    # Add Demand Constraints
    for n in demand_blocks.index:
        prob += lpSum(demand_constraints[n]) <= demand_blocks.loc[n, DEMAND_LEFT_COLUMN]

    # Add Capacity Constraints
    for m in capacity_blocks.index:
        prob += lpSum(capacity_constraints[m]) <= capacity_blocks.loc[m, CAPACITY_LEFT_COLUMN]

    return prob


def _update_blocks_df(prob: LpProblem, blocks_df: pd.DataFrame, accessibility_matrix: pd.DataFrame, accessibility: int):
    blocks_df = blocks_df.copy()

    for var in prob.variables():
        value = var.value()
        name = var.name.replace("(", "").replace(")", "").replace(",", "").split("_")
        # if name[2] == "dummy":
        #     continue
        a = int(name[1])
        b = int(name[2])
        distance = _get_distance(a, b, accessibility_matrix)
        # value = round(value)
        if value > 0:
            if distance <= accessibility:
                blocks_df.loc[a, DEMAND_WITHIN_COLUMN] += value
                blocks_df.loc[b, CAPACITY_WITHIN_COLUMN] += value
            else:
                blocks_df.loc[a, DEMAND_WITHOUT_COLUMN] += value
                blocks_df.loc[b, CAPACITY_WITHOUT_COLUMN] += value
            blocks_df.loc[a, DEMAND_LEFT_COLUMN] -= value
            blocks_df.loc[b, CAPACITY_LEFT_COLUMN] -= value

    return blocks_df


def _distribute_demand(blocks_df: pd.DataFrame, accessibility_matrix: pd.DataFrame, accessibility: int, depth: int):

    blocks_df = blocks_df.copy()
    selection_range = depth * accessibility

    prob = _set_lp_problem(blocks_df, accessibility_matrix, selection_range)

    prob.solve(PULP_CBC_CMD(msg=False))

    return _update_blocks_df(prob, blocks_df, accessibility_matrix, accessibility)


def _provision_total(blocks_df: pd.DataFrame):
    return blocks_df[DEMAND_COLUMN].sum() / blocks_df[DEMAND_COLUMN].sum()


@_validate_and_preprocess_input
def demand_based_provision(
    blocks_df: pd.DataFrame,
    accessibility_matrix: pd.DataFrame,
    demand: int,
    accessibility: int,
    self_supply: bool = True,
    max_depth: int = 1,
) -> tuple[pd.DataFrame, float]:

    blocks_df = _initialize_provision_df(blocks_df, demand)

    if self_supply:
        _supply_self(blocks_df)

    logger.info("Setting and solving an LP problem until max depth or break condition reached")
    for depth in tqdm(range(1, max_depth + 1)):
        blocks_df = _distribute_demand(blocks_df, accessibility_matrix, accessibility, depth)
        break_condition = blocks_df[DEMAND_LEFT_COLUMN].sum() == 0 or blocks_df[CAPACITY_LEFT_COLUMN].sum() == 0
        if break_condition:
            break

    blocks_df[PROVISION_COLUMN] = blocks_df[DEMAND_WITHIN_COLUMN] / blocks_df.demand
    provision_total = _provision_total(blocks_df)

    logger.success("Provision assessment finished")

    return blocks_df, provision_total
