import pandas as pd
import networkx as nx
from .common import *
from ...enums import LandUse


class LandUseConfig:
    """LandUseConfig class.

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
        """Initialize the instance.

        Parameters
        ----------
        adjacency_rules : nx.Graph, default: ADJACENCY_RULES_GRAPH
            Description.
        probability_matrix : pd.DataFrame, default: PROBABILITY_MATRIX
            Description.
        possibility_matrix : pd.DataFrame, default: POSSIBILITY_MATRIX
            Description.
        area_ranges : pd.DataFrame, default: AREA_RANGES
            Description.
        ratio_ranges : pd.DataFrame, default: RATIO_RANGES
            Description.
        fsi_ranges : pd.DataFrame, default: FSI_RANGES
            Description.
        gsi_ranges : pd.DataFrame, default: GSI_RANGES
            Description.

        Returns
        -------
        None
            Description.

        """
        self.adjacency_rules = adjacency_rules
        self.probability_matrix = probability_matrix
        self.possibility_matrix = possibility_matrix
        self.area_ranges = area_ranges
        self.ratio_ranges = ratio_ranges
        self.fsi_ranges = FSI_RANGES
        self.gsi_ranges = GSI_RANGES

    def set_adjacency_rule(self, lu_a: LandUse, lu_b: LandUse, allow: bool):
        """Set adjacency rule.

        Parameters
        ----------
        lu_a : LandUse
            Description.
        lu_b : LandUse
            Description.
        allow : bool
            Description.

        """
        allowed = self.adjacency_rules.has_edge(lu_a, lu_b)
        if allowed and not allow:
            self.adjacency_rules.remove_edge(lu_a, lu_b)
        if not allowed and allow:
            self.adjacency_rules.add_edge(lu_a, lu_b)

    def get_adjacency_rule(self, lu_a: LandUse, lu_b: LandUse) -> bool:
        """Get adjacency rule.

        Parameters
        ----------
        lu_a : LandUse
            Description.
        lu_b : LandUse
            Description.

        Returns
        -------
        bool
            Description.

        """
        return self.adjacency_rules.has_edge(lu_a, lu_b)

    def set_probability(self, lu_a: LandUse, lu_b: LandUse, probability: float):
        """Set probability.

        Parameters
        ----------
        lu_a : LandUse
            Description.
        lu_b : LandUse
            Description.
        probability : float
            Description.

        """
        if probability < 0 or probability > 1:
            raise ValueError("probability must be in range [0,1]")
        self.probability_matrix.loc[lu_a, lu_b] = probability

    def set_possibility(self, lu_a: LandUse, lu_b: LandUse, possibility: bool):
        """Set possibility.

        Parameters
        ----------
        lu_a : LandUse
            Description.
        lu_b : LandUse
            Description.
        possibility : bool
            Description.

        """
        if not isinstance(possibility, bool):
            raise ValueError("possibility must be bool")
        self.possibility_matrix.loc[lu_a, lu_b] = possibility

    def set_area_range(self, lu: LandUse, lower: float | None, upper: float | None):
        """Set area range.

        Parameters
        ----------
        lu : LandUse
            Description.
        lower : float | None
            Description.
        upper : float | None
            Description.

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
        """Set ratio range.

        Parameters
        ----------
        lu : LandUse
            Description.
        lower : float | None
            Description.
        upper : float | None
            Description.

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
