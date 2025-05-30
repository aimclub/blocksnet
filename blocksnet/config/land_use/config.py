import networkx as nx
import pandas as pd

from ...enums import LandUse
from .common import *


class LandUseConfig:
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
        allowed = self.adjacency_rules.has_edge(lu_a, lu_b)
        if allowed and not allow:
            self.adjacency_rules.remove_edge(lu_a, lu_b)
        if not allowed and allow:
            self.adjacency_rules.add_edge(lu_a, lu_b)

    def get_adjacency_rule(self, lu_a: LandUse, lu_b: LandUse) -> bool:
        return self.adjacency_rules.has_edge(lu_a, lu_b)

    def set_probability(self, lu_a: LandUse, lu_b: LandUse, probability: float):
        if probability < 0 or probability > 1:
            raise ValueError("probability must be in range [0,1]")
        self.probability_matrix.loc[lu_a, lu_b] = probability

    def set_possibility(self, lu_a: LandUse, lu_b: LandUse, possibility: bool):
        if not isinstance(possibility, bool):
            raise ValueError("possibility must be bool")
        self.possibility_matrix.loc[lu_a, lu_b] = possibility

    def set_area_range(self, lu: LandUse, lower: float | None, upper: float | None):
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
