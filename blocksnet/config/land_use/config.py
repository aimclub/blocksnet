import pandas as pd
import networkx as nx
from .common import *
from ...enums import LandUse


class LandUseConfig:
    """Store editable configuration tables for land-use modelling.

    Parameters
    ----------
    adjacency_rules : nx.Graph, default=ADJACENCY_RULES_GRAPH
        Graph describing which land-use types may border each other.
    probability_matrix : pandas.DataFrame, default=PROBABILITY_MATRIX
        Matrix of conditional probabilities for land-use transitions.
    possibility_matrix : pandas.DataFrame, default=POSSIBILITY_MATRIX
        Boolean matrix indicating whether a pair of land-use types is
        compatible.
    area_ranges : pandas.DataFrame, default=AREA_RANGES
        Allowed minimum and maximum area per land-use type.
    ratio_ranges : pandas.DataFrame, default=RATIO_RANGES
        Permissible share of each land-use type in a territory.
    fsi_ranges : pandas.DataFrame, default=FSI_RANGES
        Floor space index limits per land-use type.
    gsi_ranges : pandas.DataFrame, default=GSI_RANGES
        Ground space index limits per land-use type.
    """

    def __init__(
        self,
        adjacency_rules: nx.Graph = ADJACENCY_RULES_GRAPH,
        probability_matrix: pd.DataFrame = PROBABILITY_MATRIX,
        possibility_matrix: pd.DataFrame = POSSIBILITY_MATRIX,
        area_ranges: pd.DataFrame = AREA_RANGES,
        ratio_ranges: pd.DataFrame = RATIO_RANGES,
        fsi_ranges: pd.DataFrame = FSI_RANGES,
        gsi_ranges: pd.DataFrame = GSI_RANGES,
    ):
        self.adjacency_rules = adjacency_rules
        self.probability_matrix = probability_matrix
        self.possibility_matrix = possibility_matrix
        self.area_ranges = area_ranges
        self.ratio_ranges = ratio_ranges
        self.fsi_ranges = FSI_RANGES
        self.gsi_ranges = GSI_RANGES

    def set_adjacency_rule(self, lu_a: LandUse, lu_b: LandUse, allow: bool):
        """Update adjacency permission between two land-use types.

        Parameters
        ----------
        lu_a, lu_b : LandUse
            Land-use types that form the edge in the adjacency graph.
        allow : bool
            Whether the land-use pair should be considered adjacent.
        """

        allowed = self.adjacency_rules.has_edge(lu_a, lu_b)
        if allowed and not allow:
            self.adjacency_rules.remove_edge(lu_a, lu_b)
        if not allowed and allow:
            self.adjacency_rules.add_edge(lu_a, lu_b)

    def get_adjacency_rule(self, lu_a: LandUse, lu_b: LandUse) -> bool:
        """Return whether two land-use types are considered adjacent.

        Parameters
        ----------
        lu_a, lu_b : LandUse
            Land-use types to query.

        Returns
        -------
        bool
            ``True`` if the pair is allowed to be adjacent, ``False``
            otherwise.
        """

        return self.adjacency_rules.has_edge(lu_a, lu_b)

    def set_probability(self, lu_a: LandUse, lu_b: LandUse, probability: float):
        """Assign a conditional probability between two land-use types.

        Parameters
        ----------
        lu_a, lu_b : LandUse
            Source and target land-use types used to index the matrix.
        probability : float
            Probability value in the inclusive range ``[0, 1]``.

        Raises
        ------
        ValueError
            If *probability* is outside the valid range.
        """

        if probability < 0 or probability > 1:
            raise ValueError("probability must be in range [0,1]")
        self.probability_matrix.loc[lu_a, lu_b] = probability

    def set_possibility(self, lu_a: LandUse, lu_b: LandUse, possibility: bool):
        """Update compatibility information for two land-use types.

        Parameters
        ----------
        lu_a, lu_b : LandUse
            Land-use types used to index the possibility matrix.
        possibility : bool
            ``True`` if the transition is feasible, ``False`` otherwise.

        Raises
        ------
        ValueError
            If *possibility* is not a boolean value.
        """

        if not isinstance(possibility, bool):
            raise ValueError("possibility must be bool")
        self.possibility_matrix.loc[lu_a, lu_b] = possibility

    def set_area_range(self, lu: LandUse, lower: float | None, upper: float | None):
        """Modify permissible area range for a land-use type.

        Parameters
        ----------
        lu : LandUse
            Land-use type to configure.
        lower, upper : float or None
            Lower and upper bounds for the area. ``None`` keeps the existing
            value.

        Raises
        ------
        ValueError
            If *lower* exceeds *upper* after substitution of defaults.
        """

        if lower is None:
            lower = self.area_ranges.loc[lu, "lower"]
        if upper is None:
            upper = self.area_ranges.loc[lu, "upper"]
        if lower <= upper:
            self.area_ranges.loc[lu, "lower"] = lower
            self.area_ranges.loc[lu, "upper"] = upper
        else:
            raise ValueError(f"Lower ({lower}) must be less or equal than upper ({upper})")

    def set_ratio_range(self, lu: LandUse, lower: float | None, upper: float | None):
        """Modify acceptable share range for a land-use type.

        Parameters
        ----------
        lu : LandUse
            Land-use type to update.
        lower, upper : float or None
            Lower and upper bounds for the share. ``None`` retains the
            current configuration.

        Raises
        ------
        ValueError
            If *lower* exceeds *upper* after defaults are applied.
        """

        if lower is None:
            lower = self.ratio_ranges.loc[lu, "lower"]
        if upper is None:
            upper = self.ratio_ranges.loc[lu, "upper"]
        if lower <= upper:
            self.ratio_ranges.loc[lu, "lower"] = lower
            self.ratio_ranges.loc[lu, "upper"] = upper
        else:
            raise ValueError(f"Lower ({lower}) must be less or equal than upper ({upper})")


land_use_config = LandUseConfig()
