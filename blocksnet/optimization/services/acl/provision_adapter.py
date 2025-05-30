from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from blocksnet.analysis.provision import competitive_provision, provision_strong_total
from blocksnet.analysis.provision.competivive.core import CAPACITY_LEFT_COLUMN, DEMAND_LEFT_COLUMN, DEMAND_WITHIN_COLUMN
from blocksnet.config import log_config, service_types_config
from blocksnet.enums import LandUse
from blocksnet.optimization.services.common import ServicesContainer
from blocksnet.relations import get_accessibility_context


class ProvisionAdapter:
    """
    Adapter class for calculating and managing service provisions across urban blocks.

    This class handles service provision calculations based on:
    - Block land use characteristics
    - Service accessibility matrices
    - Service capacity distributions
    """

    def __init__(self, blocks_lus: Dict[int, LandUse], accessibility_matrix: pd.DataFrame, blocks_df: pd.DataFrame):
        """
        Initialize the ProvisionAdapter with necessary data structures.

        Parameters
        ----------
        blocks_lus : Dict[int, LandUse]
            Dictionary mapping block IDs to their land use types.
        accessibility_matrix : pd.DataFrame
            Square DataFrame representing accessibility between blocks, where index and
            columns are block IDs and values represent accessibility measures.
        blocks_df : pd.DataFrame
            DataFrame containing block geometries and attributes, indexed by block ID.
        """
        self._accessibility_matrix: pd.DataFrame = accessibility_matrix
        self._blocks_df = blocks_df.copy()
        self._blocks_df.loc[list(blocks_lus.keys()), "population"] = 0
        self._blocks_lus = blocks_lus
        self.last_provisions_dfs = {}
        self.start_provisions_dfs = {}

    def add_service_type(self, services_container: ServicesContainer) -> None:
        """
        Add a new service type and initialize its provision calculations.

        Parameters
        ----------
        services_container : ServicesContainer
            Container object holding service configuration and capacity data.

        Notes
        -----
        - Temporarily suppresses logging during calculation for cleaner output
        - Calculates initial competitive provision distribution
        - Stores resulting provision dataframe for the service type
        """
        # Store original logging settings
        log_level = log_config.logger_level
        disable_tqdm = log_config.disable_tqdm

        # Temporarily suppress logging
        log_config.set_disable_tqdm(True)
        log_config.set_logger_level("ERROR")

        blocks_ids = list(self._blocks_lus)
        services_df = services_container.services_df.copy()

        # Initialize capacities
        services_df.loc[blocks_ids, "capacity"] = 0

        # Get service type parameters
        _, demand, accessibility = service_types_config[services_container.name].values()

        # Calculate initial competitive provision
        provision_df, _ = competitive_provision(
            self._blocks_df.join(services_df), self._accessibility_matrix, accessibility, demand
        )

        # Get accessibility context for relevant blocks
        context_acc_mx = get_accessibility_context(
            self._accessibility_matrix, provision_df.loc[blocks_ids], accessibility, out=False
        )

        # Store provision data for accessible blocks
        provision_df = provision_df.loc[context_acc_mx.index]
        if provision_df.demand.sum() > 0:
            self.start_provisions_dfs[services_container.name] = provision_df
            self.last_provisions_dfs[services_container.name] = provision_df

        # Restore original logging settings
        log_config.set_disable_tqdm(disable_tqdm)
        log_config.set_logger_level(log_level)

    def get_last_provision_df(self, service_type: str) -> pd.DataFrame:
        if service_type not in self.last_provisions_dfs.keys():
            return None

        return self.last_provisions_dfs[service_type].copy()

    def get_start_provision_df(self, service_type: str) -> pd.DataFrame:
        if service_type not in self.start_provisions_dfs.keys():
            return None

        return self.start_provisions_dfs[service_type].copy()

    def calculate_provision(
        self,
        service_type: str,
        variables_df: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        Calculate the total provision score for a service type.

        Parameters
        ----------
        service_type : str
            The service type identifier to calculate provision for.
        variables_df : Optional[pd.DataFrame]
            DataFrame containing capacity updates of variables

        Returns
        -------
        float
            The total provision score (0-1) representing how well demand is met:
            - 1.0 indicates all demand is perfectly satisfied
            - Lower values indicate unmet demand
        """
        if variables_df is not None:
            # Temporarily suppress logging during updates
            log_level = log_config.logger_level
            disable_tqdm = log_config.disable_tqdm
            log_config.set_disable_tqdm(True)
            log_config.set_logger_level("ERROR")

            variables_df = variables_df[variables_df.service_type == service_type]

            # Aggregate capacity updates by block
            delta_df = variables_df.groupby("block_id").agg({"total_capacity": "sum"})

            # Get service type parameters
            _, demand, accessibility = service_types_config[service_type].values()

            # Update capacities and recalculate provision
            old_provision_df = self.get_start_provision_df(service_type)
            old_provision_df["demand"] = self.get_start_provision_df(service_type)[DEMAND_LEFT_COLUMN]

            if old_provision_df["demand"].sum() == 0:
                return 1.0

            old_provision_df["capacity"] = self.get_start_provision_df(service_type)[CAPACITY_LEFT_COLUMN]
            old_provision_df.loc[delta_df.index, "capacity"] += delta_df["total_capacity"]

            new_provision_df, _ = competitive_provision(
                old_provision_df, self._accessibility_matrix, accessibility, demand
            )

            new_provision_df["demand"] = self.get_start_provision_df(service_type)["demand"]
            new_provision_df[DEMAND_WITHIN_COLUMN] += self.get_start_provision_df(service_type)[DEMAND_WITHIN_COLUMN]

            # Restore logging and return provision score
            log_config.set_disable_tqdm(disable_tqdm)
            log_config.set_logger_level(log_level)

            self.last_provisions_dfs[service_type] = new_provision_df
            return float(provision_strong_total(new_provision_df))

        # Return current provision if no updates provided
        return float(provision_strong_total(self.get_last_provision_df(service_type)))
