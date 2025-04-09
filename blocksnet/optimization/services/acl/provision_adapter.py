from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from blocksnet.analysis.provision import competitive_provision, provision_strong_total
from blocksnet.config import log_config, service_types_config
from blocksnet.enums import LandUse
from blocksnet.optimization.services.common import ServicesContainer
from blocksnet.relations import get_accessibility_context


class ProvisionAdapter:
    """
    Adapter class for calculating and managing service provisions across urban blocks.
    
    This class handles the initialization of provision dataframes and provides methods
    to calculate service provisions based on accessibility matrices and service capacities.
    """

    def __init__(
        self,
        blocks_lus: Dict[int, LandUse],
        accessibility_matrix: pd.DataFrame,
        blocks_df: pd.DataFrame,
        services_containers: Dict[str, ServicesContainer],
    ):
        """
        Initialize the ProvisionAdapter with necessary data structures.

        Parameters
        ----------
        blocks_lus : Dict[int, LandUse]
            Dictionary mapping block IDs to their land use types.
        accessibility_matrix : pd.DataFrame
            DataFrame representing accessibility between blocks and services.
        blocks_df : pd.DataFrame
            DataFrame containing block information and characteristics.
        services_containers : Dict[str, ServicesContainer]
            Dictionary mapping service types to their respective ServicesContainer objects.
        """
        # Init provisions_dfs
        self._accessibility_matrix: pd.DataFrame = accessibility_matrix
        self.provisions_dfs = self._initialize_provisions_dfs(list(blocks_lus), blocks_df, services_containers)
        self.provisions_dfs = {
            service_type: provision_df
            for service_type, provision_df in self.provisions_dfs.items()
            if provision_df.demand.sum() > 0
        }

    def _initialize_provisions_dfs(
        self, blocks_ids: List[int], blocks_df: pd.DataFrame, services_containers: Dict[str, ServicesContainer]
    ) -> Dict[str, pd.DataFrame]:
        """
        Initialize provision dataframes for all service types.

        Parameters
        ----------
        blocks_ids : List[int]
            List of block IDs to include in the analysis.
        blocks_df : pd.DataFrame
            DataFrame containing block information and characteristics.
        services_containers : Dict[str, ServicesContainer]
            Dictionary mapping service types to their respective ServicesContainer objects.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping service types to their provision dataframes.
        """
        provisions_dfs = {}
        log_level = log_config.logger_level
        disable_tqdm = log_config.disable_tqdm

        # Temporarily suppress logging for cleaner output
        log_config.set_disable_tqdm(True)
        log_config.set_logger_level("ERROR")

        for service_type in tqdm(services_containers.keys(), disable=False):
            services_df = services_containers[service_type].services_df.copy()
            services_df.loc[blocks_ids, "capacity"] = 0
            _, demand, accessibility = service_types_config[service_type].values()

            # Calculate initial competitive provision
            provision_df, _ = competitive_provision(
                blocks_df.join(services_df), self._accessibility_matrix, accessibility, demand
            )

            # Get accessibility context for relevant blocks
            context_acc_mx = get_accessibility_context(
                self._accessibility_matrix, provision_df.loc[blocks_ids], accessibility, out=False
            )
            provisions_dfs[service_type] = provision_df.loc[context_acc_mx.index]

        # Restore original logging settings
        log_config.set_disable_tqdm(disable_tqdm)
        log_config.set_logger_level(log_level)
        return provisions_dfs

    def get_provision_df(self, service_type: str) -> pd.DataFrame:
        """
        Get the provision dataframe for a specific service type.

        Parameters
        ----------
        service_type : str
            The service type to retrieve provisions for.

        Returns
        -------
        pd.DataFrame
            DataFrame containing provision information for the specified service type.
        """
        return self.provisions_dfs[service_type]

    def calculate_provision(
        self,
        service_type: str,
        variables_df: pd.DataFrame | None = None,
    ) -> float:
        """
        Calculate the total provision for a service type, optionally updating capacities.

        Parameters
        ----------
        service_type : str
            The service type to calculate provision for.
        variables_df : pd.DataFrame | None, optional
            DataFrame containing capacity updates (block_id and total_capacity columns).
            If provided, will update capacities before calculating provision.

        Returns
        -------
        float
            The total provision score for the service type.
        """
        if variables_df is not None:
            # Aggregate capacity updates by block
            delta_df = variables_df.groupby("block_id").agg({"total_capacity": "sum"})

            # Get service type configuration
            _, demand, accessibility = service_types_config[service_type].values()
            
            # Update capacities and recalculate provision
            old_provision_df = self.provisions_dfs[service_type]
            old_provision_df.loc[delta_df.index, "capacity"] += delta_df["total_capacity"]
            new_provision_df, _ = competitive_provision(
                old_provision_df, self._accessibility_matrix, accessibility, demand
            )
            return provision_strong_total(new_provision_df)

        # Return current provision if no updates provided
        return provision_strong_total(self.get_provision_df(service_type))