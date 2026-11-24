from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from blocksnet.analysis.provision.competitive.core import DEMAND_LEFT_COLUMN
from blocksnet.config.service_types.config import service_types_config
from blocksnet.optimization.services.common.variable import Variable


class CapacityChecker:
    """
    A class for checking and managing service capacity constraints in urban block optimization problems.

    This class tracks service demand capacities, calculates upper bounds for service variables,
    and verifies that solutions don't exceed demand capacities for different service types.
    """

    def __init__(
        self, block_ids: List[int], accessibility_matrix: pd.DataFrame, neighbourhood_share: float = 1.0
    ):
        """
        Initialize the CapacityChecker with block demand data.

        Parameters
        ----------
        block_ids : List[int]
            List of block IDs to be included in the checker.
        accessibility_matrix : pd.DataFrame
            DataFrame representing accessibility between blocks, used to determine service coverage.
        neighbourhood_share : float
            How much of the demand the neighbourhood already had unmet the
            optimized blocks may take on, from 0 to 1. They always cover the
            demand of the residents they house; this is the part of everybody
            else's shortfall they are also made responsible for.

            The default is ``1``: the city around a block counts, inside the
            radius of the service in question. Note how much that is for a wide
            radius — a 60-minute service around these blocks reaches eight times
            the demand the redevelopment itself creates — so the catchment
            sharing that keeps each block from claiming all of it matters more
            the wider the radius gets.
        """
        if not 0 <= neighbourhood_share <= 1:
            raise ValueError("neighbourhood_share must be between 0 and 1")
        self._neighbourhood_share = neighbourhood_share
        self._demands: Dict[int, Dict[str, float]] = {block_id: {} for block_id in block_ids}
        self._accessibility_matrix = accessibility_matrix
        # Optimized blocks inside each block's catchment, and the reverse map.
        # A block's demand covers its whole catchment, so the blocks in that
        # catchment share it instead of each being offered the full amount.
        self._catchments: Dict[str, Dict[int, List[int]]] = {}
        self._dependents: Dict[str, Dict[int, List[int]]] = {}

    def add_service_type(self, st: str, provision_df: pd.DataFrame):
        """
        Add a service type and its provision data to the capacity checker.

        Parameters
        ----------
        st : str
            Service type identifier to be added.
        provision_df : pd.DataFrame
            DataFrame containing provision data for the service type, including demand information.
        """
        self._catchments[st] = {block_id: self.calc_catchment(block_id, st) for block_id in self._demands.keys()}
        # Whose budget a placement spends: every block whose catchment covers
        # the one built on, since they share that demand.
        dependents: Dict[int, List[int]] = {block_id: [] for block_id in self._demands.keys()}
        for block_id, catchment in self._catchments[st].items():
            for other_id in catchment:
                dependents[other_id].append(block_id)
        self._dependents[st] = dependents

        for block_id in self._demands.keys():
            block_demand = self.calc_demand(block_id, st, provision_df)
            units = service_types_config.units

            if any(units[units.service_type == st].capacity >= block_demand):
                block_demand += min(
                    [
                        unit.capacity
                        for _, unit in units[units.service_type == st].iterrows()
                        if unit.capacity >= block_demand
                    ]
                )
            else:
                block_demand += max(
                    [
                        unit.capacity
                        for _, unit in units[units.service_type == st].iterrows()
                        if unit.capacity <= block_demand
                    ]
                )

            self._demands[block_id][st] = max(block_demand, 0)

    def calc_demand(self, block_id: int, service: str, provision_df: pd.DataFrame) -> float:
        """
        Calculate the total demand for a service in a given block's accessible area.

        Parameters
        ----------
        block_id : int
            ID of the block to calculate demand for.
        service : str
            Service type identifier.
        provision_df : pd.DataFrame
            DataFrame containing provision data for the service.

        Returns
        -------
        float
            Total calculated demand for the service in the accessible area around the block.
        """
        # Get accessibility threshold for this service type
        _, _, accessibility = service_types_config[service].values()

        # Find blocks within accessibility range
        nearest_blocks = self._accessibility_matrix.index[self._accessibility_matrix[block_id] <= accessibility]

        return provision_df[provision_df.index.isin(nearest_blocks)][DEMAND_LEFT_COLUMN].sum()

    def calc_catchment(self, block_id: int, service: str) -> List[int]:
        """
        List the optimized blocks inside the accessibility catchment of a block.

        Parameters
        ----------
        block_id : int
            ID of the block whose catchment is requested.
        service : str
            Service type identifier.

        Returns
        -------
        List[int]
            Optimized block IDs inside the catchment of ``block_id``.
        """
        _, _, accessibility = service_types_config[service].values()
        nearest_blocks = set(self._accessibility_matrix.index[self._accessibility_matrix[block_id] <= accessibility])
        return [other_id for other_id in self._demands.keys() if other_id in nearest_blocks]

    def get_catchment(self, block_id: int, service: str) -> List[int]:
        """
        Get the optimized blocks sharing the demand of a block.

        Parameters
        ----------
        block_id : int
            ID of the block whose catchment is requested.
        service : str
            Service type identifier.

        Returns
        -------
        List[int]
            Block IDs sharing the demand of ``block_id``, or just the block
            itself if the service type is unknown.
        """
        return self._catchments.get(service, {}).get(block_id, [block_id])

    def get_dependents(self, block_id: int, service: str) -> List[int]:
        """
        Get the blocks whose demand is consumed by building in a block.

        Parameters
        ----------
        block_id : int
            ID of the block services would be placed in.
        service : str
            Service type identifier.

        Returns
        -------
        List[int]
            Blocks whose catchment contains ``block_id``, or just the block
            itself if the service type is unknown.
        """
        return self._dependents.get(service, {}).get(block_id, [block_id])

    def get_catchment_demand(
        self, block_id: int, service: str, new_residents_demand: Optional[Dict[int, Dict[str, float]]] = None
    ) -> float:
        """
        Demand of a block's catchment: the existing one plus its new residents.

        ``_demands`` only holds the demand that existed before the blocks were
        redeveloped. Residents the solution houses generate demand of their own,
        and it belongs to the same catchment pool the services are compared
        against; leaving it out means a block may not serve the people it just
        housed.

        Parameters
        ----------
        block_id : int
            ID of the block whose catchment is requested.
        service : str
            Service type identifier.
        new_residents_demand : Optional[Dict[int, Dict[str, float]]]
            Demand generated by new residents, per block and service type.

        Returns
        -------
        float
            Total demand inside the catchment of ``block_id``.
        """
        # Everyone the service can reach: what the city around still lacks and
        # what the redeveloped blocks nearby will need, both taken inside this
        # service's own radius. Pooling the redevelopment's demand across all
        # its blocks regardless of distance let a block be served by a facility
        # an hour's walk away.
        demand = self._neighbourhood_share * self._demands[block_id][service]
        if new_residents_demand:
            demand += sum(
                new_residents_demand.get(other_id, {}).get(service, 0)
                for other_id in self.get_catchment(block_id, service)
            )
        return demand

    def get_demand(self, block_id: int, service: str, demand: int = 0) -> float:
        """
        Retrieve the demand for a specific service in a block, optionally adjusted by additional population.

        Parameters
        ----------
        block_id : int
            ID of the block to get demand for.
        service : str
            Service type identifier.
        demand : int, optional
            Additional demand to add to the base demand (default is 0).

        Returns
        -------
        float
            Total demand for the service in the block, including any additional population.
        """
        return self._neighbourhood_share * self._demands[block_id][service] + demand

    def get_ub_var(self, var: Variable, demand: int = 0) -> int:
        """
        Calculate the upper bound (maximum count) for a service variable based on remaining demand capacity.

        Parameters
        ----------
        var : Variable
            The service variable to calculate upper bound for.
        demand : int, optional
            Additional demand to consider in demand calculation (default is 0).

        Returns
        -------
        int
            Maximum possible count for the variable given remaining demand capacity.
            Returns -1 if the variable has zero capacity.
        """
        block_demand = self.get_demand(var.block_id, var.service_type, demand)
        if var.capacity == 0:
            return -1
        return int(np.floor(block_demand / var.capacity))

    def check_constraints(
        self, X: List[Variable], blocks_services_demand: Optional[Dict[int, Dict[str, int]]] = None
    ) -> bool:
        """
        Verify that the solution doesn't exceed demand capacity constraints for any service type.

        Parameters
        ----------
        X : List[Variable]
            List of Variable objects representing the solution.
        blocks_services_demand : Optional[Dict[int, Dict[str, int]]], optional
            Additional demand per service type per block to consider in constraints (default is None).

        Returns
        -------
        bool
            True if all service capacities in the solution satisfy demand constraints,
            False if any service exceeds its demand capacity.
        """
        placed: Dict[int, Dict[str, float]] = {}
        for var in X:
            placed.setdefault(var.block_id, {}).setdefault(var.service_type, 0.0)
            placed[var.block_id][var.service_type] += var.total_capacity

        # One check, per catchment: what a service can reach against what needs
        # serving there. A ceiling over the redevelopment as a whole was also
        # tried and removed — summing demand over blocks double counts when
        # their catchments overlap and cuts legitimate plans when they do not.
        for block_id, per_block in self._demands.items():
            for service in per_block:
                served = sum(
                    placed.get(other_id, {}).get(service, 0.0) for other_id in self.get_catchment(block_id, service)
                )
                if served > self.get_catchment_demand(block_id, service, blocks_services_demand):
                    return False

        return True
