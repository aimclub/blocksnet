import random
import math
import shapely
import geopandas as gpd
import pandas as pd
import networkx as nx
from tqdm import tqdm
from shapely.ops import split
from ..models.land_use import LandUse
from ..models.schema import BaseSchema
from ..utils.helpers import get_polygon_aspect_ratio


class BlocksSchema(BaseSchema):
    """
    Schema for validating blocks GeoDataFrame.

    Attributes
    ----------
    _geom_types : list
        List of allowed geometry types for the blocks, default is [shapely.Polygon].
    """

    _geom_types = [shapely.Polygon]


ADJACENCY_RULES = [
    # self adjacency
    (LandUse.RESIDENTIAL, LandUse.RESIDENTIAL),
    (LandUse.BUSINESS, LandUse.BUSINESS),
    (LandUse.RECREATION, LandUse.RECREATION),
    (LandUse.INDUSTRIAL, LandUse.INDUSTRIAL),
    (LandUse.TRANSPORT, LandUse.TRANSPORT),
    (LandUse.SPECIAL, LandUse.SPECIAL),
    (LandUse.AGRICULTURE, LandUse.AGRICULTURE),
    # recreation can be adjacent to anything
    (LandUse.RECREATION, LandUse.SPECIAL),
    (LandUse.RECREATION, LandUse.INDUSTRIAL),
    (LandUse.RECREATION, LandUse.BUSINESS),
    (LandUse.RECREATION, LandUse.AGRICULTURE),
    (LandUse.RECREATION, LandUse.TRANSPORT),
    (LandUse.RECREATION, LandUse.RESIDENTIAL),
    # residential
    (LandUse.RESIDENTIAL, LandUse.BUSINESS),
    # business
    (LandUse.BUSINESS, LandUse.INDUSTRIAL),
    (LandUse.BUSINESS, LandUse.TRANSPORT),
    # industrial
    (LandUse.INDUSTRIAL, LandUse.SPECIAL),
    (LandUse.INDUSTRIAL, LandUse.AGRICULTURE),
    (LandUse.INDUSTRIAL, LandUse.TRANSPORT),
    # transport
    (LandUse.TRANSPORT, LandUse.SPECIAL),
    (LandUse.TRANSPORT, LandUse.AGRICULTURE),
    # special
    (LandUse.SPECIAL, LandUse.AGRICULTURE),
]

RULES_GRAPH = nx.from_edgelist(ADJACENCY_RULES)

AREA_RANGES = {
    LandUse.RESIDENTIAL: (2_000, 100_000),
    LandUse.BUSINESS: (50_000, 150_000),
    LandUse.RECREATION: (10_000, 1_000_000),
    LandUse.SPECIAL: (50_000, 500_000),
    LandUse.INDUSTRIAL: (10_000, 800_000),
    LandUse.AGRICULTURE: (300_000, 1_000_000),
    LandUse.TRANSPORT: (50_000, 500_000),
}

ASPECT_RATIO_RANGES = {
    LandUse.RESIDENTIAL: (1, 3),
    LandUse.BUSINESS: (1, 4),
    LandUse.RECREATION: (1, 7),
    LandUse.SPECIAL: (1, 6),
    LandUse.INDUSTRIAL: (1, 5),
    LandUse.AGRICULTURE: (1, 4),
    LandUse.TRANSPORT: (1, 7),
}

BUFFER_MIN = 10
BUFFER_STEP = 0.5


class LandUseOptimizer:
    """
    Optimizes the allocation of land use within a set of urban blocks.

    Parameters
    ----------
    blocks : gpd.GeoDataFrame
        GeoDataFrame containing the urban blocks with geometry.
    verbose : bool, optional
        If True, progress is displayed using tqdm (default is True).
    """

    def __init__(self, blocks: gpd.GeoDataFrame, verbose: bool = True):
        """
        Initializes the LandUseOptimizer class by validating and processing the blocks data.

        Parameters
        ----------
        blocks : gpd.GeoDataFrame
            The GeoDataFrame representing the urban blocks.
        verbose : bool, optional
            Whether to display progress with tqdm (default is True).
        """
        blocks = BlocksSchema(blocks)
        # while blocks.geometry.apply(self._is_block_large).any():
        #     blocks = self._split_large_blocks(blocks)
        self.blocks = blocks
        self.adjacency_graph = self._get_adjacency_graph(blocks)
        self.verbose = verbose

    # @staticmethod
    # def _get_possible_lus(polygon) -> list[LandUse]:
    #     ar = get_polygon_aspect_ratio(polygon)
    #     area = polygon.area
    #     lus = []
    #     for lu in list(LandUse):
    #         min_area, max_area = AREA_RANGES[lu]
    #         min_ar, max_ar = ASPECT_RATIO_RANGES[lu]
    #         if ar>=min_ar and ar<=max_ar and area>=min_area and area<=max_area:
    #             lus.append(lu)
    #     return lus

    @staticmethod
    def _split_polygon(polygon) -> shapely.GeometryCollection:
        """
        Splits a polygon into two parts by cutting along its midpoint.

        Parameters
        ----------
        polygon : shapely.Polygon
            The polygon to split.

        Returns
        -------
        shapely.GeometryCollection
            A collection of geometries resulting from the split operation.
        """
        min_rect = polygon.buffer(1).minimum_rotated_rectangle
        rect_coords = list(min_rect.exterior.coords)

        side_lengths = [
            ((rect_coords[i][0] - rect_coords[i - 1][0]) ** 2 + (rect_coords[i][1] - rect_coords[i - 1][1]) ** 2) ** 0.5
            for i in range(1, 5)
        ]
        length_1, length_2 = side_lengths[0], side_lengths[1]
        if length_1 >= length_2:
            long_side_1 = [rect_coords[0], rect_coords[1]]
            long_side_2 = [rect_coords[2], rect_coords[3]]
        else:
            long_side_1 = [rect_coords[1], rect_coords[2]]
            long_side_2 = [rect_coords[3], rect_coords[0]]

        # Рассчитываем середину длинной стороны
        midpoint_1 = ((long_side_1[0][0] + long_side_1[1][0]) / 2, (long_side_1[0][1] + long_side_1[1][1]) / 2)
        midpoint_2 = ((long_side_2[0][0] + long_side_2[1][0]) / 2, (long_side_2[0][1] + long_side_2[1][1]) / 2)
        cutting_line = shapely.LineString([midpoint_1, midpoint_2])

        return split(polygon, cutting_line)

    @staticmethod
    def _is_block_large(block_geometry: shapely.Polygon) -> bool:
        """
        Determines if a block is too large based on its area and aspect ratio.

        Parameters
        ----------
        block_geometry : shapely.Polygon
            The geometry of the block to evaluate.

        Returns
        -------
        bool
            True if the block exceeds size or aspect ratio constraints, False otherwise.
        """
        area = block_geometry.area
        aspect_ratio = get_polygon_aspect_ratio(block_geometry)
        lus = []
        for lu in list(LandUse):
            max_area = AREA_RANGES[lu][1]
            max_aspect_ratio = ASPECT_RATIO_RANGES[lu][1]
            if area <= max_area and aspect_ratio <= max_aspect_ratio:
                lus.append(lu)
        return len(lus) == 0

    @classmethod
    def _split_large_blocks(cls, blocks_gdf: gpd.GeoDataFrame):
        """
        Splits blocks in the GeoDataFrame that are too large according to predefined rules.

        Parameters
        ----------
        blocks_gdf : gpd.GeoDataFrame
            The GeoDataFrame containing block geometries.

        Returns
        -------
        gpd.GeoDataFrame
            Updated GeoDataFrame with large blocks split into smaller ones.
        """
        blocks_gdf = blocks_gdf.copy()
        blocks_gdf.geometry = blocks_gdf.geometry.apply(
            lambda g: cls._split_polygon(g) if cls._is_block_large(g) else g
        )
        return blocks_gdf.explode(index_parts=True).reset_index(drop=True)

    @classmethod
    def _get_adjacency_graph(cls, blocks_gdf: gpd.GeoDataFrame, buffer: float = BUFFER_MIN):
        """
        Generates an adjacency graph of blocks based on their spatial proximity.

        Parameters
        ----------
        blocks_gdf : gpd.GeoDataFrame
            The GeoDataFrame of blocks with geometries.
        buffer : float, optional
            Buffer size used for detecting intersections (default is BUFFER_MIN).

        Returns
        -------
        nx.Graph
            A NetworkX graph representing the adjacency relations between blocks.
        """
        blocks_gdf = blocks_gdf.copy()
        blocks_gdf.geometry = blocks_gdf.buffer(buffer)
        sjoin = gpd.sjoin(blocks_gdf, blocks_gdf, predicate="intersects")
        edge_list = [(i, s.index_right) for i, s in sjoin[sjoin.index != sjoin.index_right][["index_right"]].iterrows()]
        if len(edge_list) == 0:
            return cls._get_adjacency_graph(blocks_gdf, buffer + BUFFER_STEP)
        return nx.from_edgelist(edge_list)

    def _generate_initial_X(self) -> dict[int, LandUse]:
        """
        Generates an initial configuration of land uses for the blocks.

        Returns
        -------
        dict[int, LandUse]
            A dictionary mapping block IDs to initial land uses (default is LandUse.RECREATION).
        """
        return {block_id: LandUse.RECREATION for block_id in self.blocks.index}

    @staticmethod
    def _perturb(X: dict[int, LandUse]) -> dict[int, LandUse]:
        """
        Perturbs the current land use configuration by randomly changing the land use of one block.

        Parameters
        ----------
        X : dict[int, LandUse]
            Current land use configuration.

        Returns
        -------
        dict[int, LandUse]
            Modified configuration with one randomly changed land use.
        """
        X = {block_id: lu for block_id, lu in X.items()}
        b_id = random.choice(list(X.keys()))
        x_lu = X[b_id]
        possible_lus = list(LandUse)
        possible_lus.remove(x_lu)
        X[b_id] = random.choice(possible_lus)
        return X

    def _check_adj_rules(self, X: dict[int, LandUse]) -> bool:
        """
        Validates the adjacency rules for the given land use configuration.

        Parameters
        ----------
        X : dict[int, LandUse]
            Land use configuration to check.

        Returns
        -------
        bool
            True if the configuration satisfies adjacency rules, False otherwise.
        """
        for u, v in self.adjacency_graph.edges:
            if not RULES_GRAPH.has_edge(X[u], X[v]):
                return False
        return True

    def _check_area_ranges(self, X: dict[int, LandUse]):
        """
        Validates that each block's area falls within the allowed range for its land use.

        Parameters
        ----------
        X : dict[int, LandUse]
            Land use configuration to check.

        Returns
        -------
        bool
            True if all areas are within valid ranges, False otherwise.
        """
        for b_id, lu in X.items():
            area_min, area_max = AREA_RANGES[lu]
            block_area = self.blocks.loc[b_id, "geometry"].area
            if block_area < area_min or block_area > area_max:
                return False
        return True

    def _check_ratio_ranges(self, X: dict[int, LandUse]):
        """
        Checks whether the aspect ratio of each block falls within the allowed range for its land use.

        Parameters
        ----------
        X : dict[int, LandUse]
            Land use configuration to check.

        Returns
        -------
        bool
            True if all aspect ratios are valid, False otherwise.
        """
        for b_id, lu in X.items():
            ar = get_polygon_aspect_ratio(self.blocks.loc[b_id, "geometry"])
            min_ar, max_ar = ASPECT_RATIO_RANGES[lu]
            if ar < min_ar or ar > max_ar:
                return False
        return True

    def to_shares_dict(self, X: dict[int, LandUse]) -> pd.DataFrame:
        """
        Converts the land use configuration to a dictionary of land use area shares.

        Parameters
        ----------
        X : dict[int, LandUse]
            Land use configuration.

        Returns
        -------
        pd.DataFrame
            A dictionary mapping each land use to its share of the total area.
        """
        gdf = self.to_gdf(X)
        total_area = gdf.area.sum()
        return {lu: gdf[gdf.land_use == lu.value].area.sum() / total_area for lu in list(LandUse)}

    def to_gdf(self, X: dict[int, LandUse]) -> gpd.GeoDataFrame:
        """
        Converts the land use configuration into a GeoDataFrame with land use labels.

        Parameters
        ----------
        X : dict[int, LandUse]
            Land use configuration.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with land use assignments.
        """
        gdf = self.blocks.copy()
        gdf["land_use"] = gdf.apply(lambda s: X[s.name].value, axis=1)
        return gdf

    def _objective(self, X: dict[int, LandUse], lu_shares: dict[LandUse, float]) -> float:
        """
        Computes the objective function value, which measures deviation from target land use shares.

        Parameters
        ----------
        X : dict[int, LandUse]
            Land use configuration.
        lu_shares : dict[LandUse, float]
            Target land use shares.

        Returns
        -------
        float
            Sum of squared deviations between actual and target land use shares.
        """
        actual_shares = self.to_shares_dict(X)
        deviations = []
        for lu, share in lu_shares.items():
            actual_share = actual_shares[lu]
            deviation = (actual_share - share) ** 2
            deviations.append(deviation)
        return sum(deviations)

    def run(
        self,
        lu_shares: dict[LandUse, float],
        rate: float = 0.99,
        t_max: float = 100,
        t_min: float = 1e-3,
        max_iter=10000,
    ) -> tuple:
        """
        Runs the optimization algorithm to find the optimal land use configuration.

        Parameters
        ----------
        lu_shares : dict[LandUse, float]
            Target shares for each land use, must sum to 1.
        rate : float, optional
            Cooling rate for the simulated annealing algorithm (default is 0.99).
        t_max : float, optional
            Initial temperature for simulated annealing (default is 100).
        t_min : float, optional
            Minimum temperature for simulated annealing (default is 1e-3).
        max_iter : int, optional
            Maximum number of iterations for the optimization (default is 10000).

        Returns
        -------
        tuple
            A tuple containing the best land use configuration, best objective value, list of configurations, and list of objective values.
        """

        assert (
            round(sum(lu_shares.values()), 3) == 1
        ), f"LandUse shares sum must be  equal 1, got {sum(lu_shares.values())}"

        best_X = self._generate_initial_X()
        best_value = self._objective(best_X, lu_shares)
        T = t_max

        Xs = []
        values = []

        current_X = best_X
        current_value = best_value

        if self.verbose:
            pbar = tqdm(range(max_iter))

        for _ in range(max_iter):

            if self.verbose:
                pbar.update(1)
                pbar.set_description(f"Value : {round(current_value, 3)}")

            Xs.append(current_X)
            values.append(current_value)

            # Генерируем новое решение
            X = self._perturb(current_X)

            # Проверка ограничений
            if not self._check_adj_rules(X):
                continue
            # if not self._check_area_ranges(X):
            #   continue
            # if not self._check_ratio_ranges(X):
            #     continue

            # Вычисляем значение целевой функции
            value = self._objective(X, lu_shares)

            # Если новое решение лучше, принимаем его
            if value < current_value:
                current_value = value
                current_X = X
                # Если оно еще лучше лучшего, то меняем и его
                if value < best_value:
                    best_value = value
                    best_X = X
            else:
                # Принимаем худшее решение с вероятностью, зависящей от температуры
                delta = current_value - value
                if random.random() < math.exp(delta / T):
                    current_value = value
                    current_X = X

            # Охлаждаем температуру
            T = T * rate
            if T < t_min:
                break

        return best_X, best_value, Xs, values
