import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from loguru import logger
from blocksnet.enums import LandUse
from blocksnet.analysis.diversity.shannon.core import shannon_diversity, SHANNON_DIVERSITY_COLUMN, COUNT_COLUMN
from .schemas import BlocksSchema

DENSITY_COLUMN = "density"
LU_CONST_COLUMN = "lu_const"
ATTRACTIVENESS_COLUMN = "attractiveness"
POPULATION_COLUMN = "population"

LU_CONSTS = {
    LandUse.INDUSTRIAL: 0.25,
    LandUse.BUSINESS: 0.3,
    LandUse.SPECIAL: 0.1,
    LandUse.TRANSPORT: 0.1,
    LandUse.RESIDENTIAL: 0.1,
    LandUse.AGRICULTURE: 0.05,
    LandUse.RECREATION: 0.05,
}
DEFAULT_LU_CONST = 0.06

LU_TRIP_RATES = {
    LandUse.RESIDENTIAL: 1.0,
    LandUse.BUSINESS: 2.7,
    LandUse.INDUSTRIAL: 2.0,
    LandUse.SPECIAL: 1.2,
    LandUse.TRANSPORT: 1.0,
    LandUse.RECREATION: 1.4,
    LandUse.AGRICULTURE: 0.2,
}
DEFAULT_TRIP_RATE = 1.0

DEFAULT_ACCESSIBILITY = 10


def _round_probabilistic_row_to_int(prob_row: pd.Series, trips: int) -> pd.Series:

    # Convert one probability row into integer trips preserving row sum exactly.

    raw = prob_row.to_numpy(dtype=float) * float(trips)
    floored = np.floor(raw).astype("int64")
    remainder = int(trips - floored.sum())

    if remainder > 0:
        fractional = raw - floored
        order = np.argsort(-fractional, kind="mergesort")
        floored[order[:remainder]] += 1

    return pd.Series(floored, index=prob_row.index, dtype="int64")


def _integerize_origin_constrained_od(od_prob_mx: pd.DataFrame, demand: pd.Series) -> pd.DataFrame:

    # Integerize origin-constrained OD probabilities preserving integer row demand.

    od_int = pd.DataFrame(0, index=od_prob_mx.index, columns=od_prob_mx.columns, dtype="int64")
    demand_int = demand.round().clip(lower=0).astype("int64")

    for origin, trips in demand_int.items():
        trips = int(trips)
        if trips <= 0:
            continue

        prob_row = od_prob_mx.loc[origin].astype(float)
        row_sum = float(prob_row.sum())

        if row_sum <= 0.0:
            prob_row[:] = 0.0
            if origin in prob_row.index:
                prob_row.loc[origin] = 1.0
        else:
            prob_row = prob_row / row_sum

        od_int.loc[origin] = _round_probabilistic_row_to_int(prob_row, trips)

    return od_int


def _calculate_nodes_weights(
    blocks_df: gpd.GeoDataFrame,
    acc_mx: pd.DataFrame,
    accessibility: float,
    trip_rates: dict[LandUse, float],
) -> pd.DataFrame:

    logger.info("Identifying nearest nodes to blocks")
    acc_mx = acc_mx.replace(0, 0.1)
    acc_mask = acc_mx <= accessibility
    acc_mask = acc_mask | acc_mx.eq(acc_mx.min(axis=1), axis=0)

    logger.info("Calculating weights")
    weights_mx = pd.DataFrame(0.0, index=acc_mx.index, columns=acc_mx.columns)
    weights_mx[acc_mask] = 1.0 / acc_mx[acc_mask]
    weights_sum = weights_mx.sum(axis=1)
    weights_mx = weights_mx.div(weights_sum, axis=0)

    effective_population = blocks_df[POPULATION_COLUMN] * blocks_df.land_use.map(
        lambda lu: trip_rates.get(lu, DEFAULT_TRIP_RATE)
    )

    logger.info("Distributing")
    nodes_df = pd.DataFrame(index=acc_mx.columns)
    nodes_df[ATTRACTIVENESS_COLUMN] = weights_mx.mul(blocks_df[ATTRACTIVENESS_COLUMN], axis=0).sum(axis=0)
    nodes_df[POPULATION_COLUMN] = weights_mx.mul(effective_population, axis=0).sum(axis=0)
    return nodes_df


def _calculate_diversity(blocks_df: pd.DataFrame, services_count_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    logger.info("Calculating diversity and density")
    diversity_df = shannon_diversity(services_count_dfs)
    blocks_df = blocks_df.join(diversity_df)
    blocks_df[DENSITY_COLUMN] = blocks_df[COUNT_COLUMN] / blocks_df.site_area
    return blocks_df


def _calculate_attractiveness(blocks_df: pd.DataFrame, lu_consts: dict[LandUse, float]) -> pd.DataFrame:
    logger.info("Calculating attractiveness")
    blocks_df = blocks_df.copy()
    blocks_df[LU_CONST_COLUMN] = blocks_df.land_use.apply(lambda lu: lu_consts.get(lu, DEFAULT_LU_CONST))
    scaler = MinMaxScaler()
    columns = [DENSITY_COLUMN, SHANNON_DIVERSITY_COLUMN, LU_CONST_COLUMN]
    blocks_df[columns] = scaler.fit_transform(blocks_df[columns])
    blocks_df[ATTRACTIVENESS_COLUMN] = (
        blocks_df[DENSITY_COLUMN] + blocks_df[SHANNON_DIVERSITY_COLUMN] + blocks_df[LU_CONST_COLUMN]
    )
    return blocks_df


def _calculate_od_mx(nodes_df: pd.DataFrame, acc_mx: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating origin destination matrix")
    acc_mx = acc_mx.replace(0, np.nan)

    # Origin-constrained gravity model:
    # row sums match origin population and total OD matches total population.
    gravity_weights = (1.0 / acc_mx).mul(nodes_df[ATTRACTIVENESS_COLUMN], axis=1).fillna(0.0)

    # Preserve demand for isolated origins as intrazonal flow.
    empty_rows = gravity_weights.sum(axis=1).eq(0.0)
    for node_id in gravity_weights.index[empty_rows]:
        gravity_weights.loc[node_id, node_id] = 1.0

    row_sums = gravity_weights.sum(axis=1).replace(0.0, np.nan)
    od_prob_mx = gravity_weights.div(row_sums, axis=0).fillna(0.0)

    return _integerize_origin_constrained_od(od_prob_mx, nodes_df[POPULATION_COLUMN])


def _validate_input(blocks_df: pd.DataFrame, blocks_to_nodes_mx: pd.DataFrame, nodes_to_nodes_mx: pd.DataFrame):
    logger.info("Validating input data")
    if not all(blocks_df.index == blocks_to_nodes_mx.index):
        raise ValueError("blocks_df index and blocks_to_nodes_mx index must match")
    if not all(blocks_to_nodes_mx.columns == nodes_to_nodes_mx.index):
        raise ValueError("blocks_to_nodes_mx columns and nodes_to_nodes index must match")
    if not all(nodes_to_nodes_mx.index == nodes_to_nodes_mx.columns):
        raise ValueError("nodes_to_nodes_mx index and columns must match")


def origin_destination_matrix(
    blocks_df: pd.DataFrame,
    blocks_to_nodes_mx: pd.DataFrame,
    nodes_to_nodes_mx: pd.DataFrame,
    services_count_dfs: list[pd.DataFrame],
    accessibility: float = DEFAULT_ACCESSIBILITY,
    lu_consts: dict[LandUse, float] = LU_CONSTS,
    lu_trip_rates: dict[LandUse, float] = LU_TRIP_RATES,
) -> pd.DataFrame:
    
    """
    Build an origin-constrained OD matrix from block attributes and accessibility matrices.

    The function estimates an integer-valued origin-destination (OD) matrix between network nodes
    using an origin-constrained gravity model. Trip productions are derived from block population
    (after distributing blocks to nearby nodes), while trip attractions are proportional to a
    composite attractiveness score based on service density, service diversity (Shannon index),
    and a land-use constant.

    Parameters
    ----------
    blocks_df : pandas.DataFrame
        Table of blocks (zones). Index must uniquely identify blocks and must match
        ``blocks_to_nodes_mx.index``. Required columns are validated/coerced by ``BlocksSchema``.
        Expected fields include at least:
        - ``population`` (number of people living in block, a field mapped to :data:`POPULATION_COLUMN`)
        - ``land_use`` (type of land use for urban block, values of :class:`blocksnet.enums.LandUse`)
        - ``site_area`` (used for service density)
    blocks_to_nodes_mx : pandas.DataFrame
        Block-to-node generalized cost matrix (e.g., walk time), with blocks on rows and network
        nodes on columns. Used to distribute each block's population/attractiveness across nearby
        nodes. Its index must match ``blocks_df.index``.
    nodes_to_nodes_mx : pandas.DataFrame
        Node-to-node generalized cost matrix (e.g., travel time). Must be square with identical
        index/columns, and its index must match ``blocks_to_nodes_mx.columns``.
        Zeros are treated as missing costs for the gravity model.
    services_count_dfs : list[pandas.DataFrame]
        List of service-count tables (typically one per service category/type) used to compute
        Shannon diversity and total service counts per block. Each table must be compatible with
        :func:`blocksnet.analysis.diversity.shannon.core.shannon_diversity`.
    accessibility : float, default=DEFAULT_ACCESSIBILITY
        Threshold on ``blocks_to_nodes_mx`` costs defining which nodes are considered reachable
        from a block for distribution. For each block, nodes with cost ``<= accessibility`` are
        considered; additionally, the nearest node is always included to avoid empty supports.
    lu_consts : dict[LandUse, float], default=LU_CONSTS
        Mapping from land-use type to an additive attractiveness constant. If a block land-use is
        missing in the mapping, :data:`DEFAULT_LU_CONST` is used.

    Returns
    -------
    pandas.DataFrame
        Integer OD matrix between network nodes. Index and columns correspond to
        ``nodes_to_nodes_mx.index``. Row sums are equal to the corresponding origin node population
        (rounded to integers and clipped at zero). Total trips equal the total distributed
        population (after rounding).

    Notes
    -----
    - This is an *origin-constrained* model: productions (row sums) are fixed by origin population,
      while attractions (column sums) are not explicitly constrained.
    - Blocks are not OD zones directly: block attributes are first distributed to network nodes
      via inverse-cost weights derived from ``blocks_to_nodes_mx``. The OD matrix is then computed
      between nodes.
    - If an origin node has no valid outgoing gravity weights (e.g., all costs missing), its demand
      is assigned as intrazonal flow (diagonal element).
    - Integerization preserves each origin row sum exactly using largest-remainder rounding.

    Raises
    ------
    ValueError
        If indices/columns of the input matrices are inconsistent (do not align), or if the block
        table does not satisfy the expected schema after validation.

    Examples
    --------
    >>> od_nodes = origin_destination_matrix(
    ...     blocks_df=blocks,
    ...     blocks_to_nodes_mx=blocks_to_nodes,
    ...     nodes_to_nodes_mx=nodes_to_nodes,
    ...     services_count_dfs=[shops_counts, schools_counts],
    ...     accessibility=10.0,
    ... )
    >>> od_nodes.shape
    (len(nodes_to_nodes.index), len(nodes_to_nodes.index))
    """

    blocks_df = BlocksSchema(blocks_df)
    _validate_input(blocks_df, blocks_to_nodes_mx, nodes_to_nodes_mx)

    blocks_df = _calculate_diversity(blocks_df, services_count_dfs)
    blocks_df = _calculate_attractiveness(blocks_df, lu_consts)

    nodes_gdf = _calculate_nodes_weights(blocks_df, blocks_to_nodes_mx, accessibility, lu_trip_rates)

    return _calculate_od_mx(nodes_gdf, nodes_to_nodes_mx)
