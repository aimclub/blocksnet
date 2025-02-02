from typing import Dict, List

import pandas as pd

from blocksnet import City, LandUse, Provision, ServiceType
from blocksnet.method.annealing_optimizer import Indicator, Variable


class ProvisionAdapter:
    """
    Class that adapts the calculation of provisions based on the current solution variables.

    This class provides methods to calculate and update provisions for various service types in the city model,
    using the `Provision` instance to compute values based on a given set of variables and indicators.
    """

    def __init__(self, city_model: City, blocks_lu: Dict[int, LandUse]):
        """
        Initializes the ProvisionAdapter with the provided city model and land use by block information.

        Parameters
        ----------
        city_model : City
            The city model instance that contains the blocks and service types.
        blocks_lu : Dict[int, LandUse]
            A dictionary mapping block IDs to their corresponding land use types.
        """
        self._provision_instance: Provision = Provision(city_model=city_model, verbose=False)
        self._clear_df: pd.DataFrame = self._get_clear_df(blocks_lu.keys())  # Initialize clear dataframe

    def _vars_to_df(self, X: List[Variable], indicators: Dict[int, Indicator]) -> pd.DataFrame:
        """
        Converts the solution variables into a DataFrame format for provision calculation.

        This method generates a DataFrame containing the block IDs, population values, and service capacities
        based on the solution variables and their indicators.

        Parameters
        ----------
        X : List[Variable]
            List of variables representing the current solution.
        indicators : Dict[int, Indicator]
            Dictionary mapping block IDs to their corresponding indicators.

        Returns
        -------
        pd.DataFrame
            A DataFrame with population and service capacities for each block.
        """
        service_types = {x.service_type for x in X}  # Collect all service types from the variables
        df = pd.DataFrame(
            [
                {
                    "block_id": x.block.id,
                    "population": indicators[x.block.id].population,  # Population for each block
                    x.service_type.name: x.capacity,  # Service capacity for each type
                }
                for x in X
            ]
        )
        # Group the DataFrame by block_id and aggregate service capacities
        return df.groupby("block_id").agg({"population": "min", **{st.name: "sum" for st in service_types}})

    def _get_clear_df(self, blocks: List[int]) -> pd.DataFrame:
        """
        Generates a DataFrame representing the clear state of provisions for a given set of blocks.

        This method creates an initial DataFrame that contains the negative population values and
        capacity values for each service type.

        Parameters
        ----------
        blocks : List[int]
            List of block IDs for which to generate the initial DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame representing the clear state of provisions for the given blocks.
        """
        gdf = self._provision_instance.city_model.get_blocks_gdf()
        gdf = gdf[gdf.index.isin(blocks)]  # Filter blocks based on the provided block IDs
        df = gdf[["population"]].copy()
        df["population"] = -df["population"]
        for column in [column for column in gdf.columns if "capacity_" in column]:
            st_name = column.removeprefix("capacity_")
            df[st_name] = -gdf[column]
        return df

    def calculate_provision(
        self, X: List[Variable], service_type: ServiceType, indicators: Dict[int, Indicator]
    ) -> float:
        """
        Calculate the provision for a specific service type.

        This method computes the provision for a given service type by updating the clear DataFrame with
        the current solution variables and calculating the resulting provision value.

        Parameters
        ----------
        X : list[Variable]
            List of variables representing the current solution.
        service_type : ServiceType
            The service type to calculate provision for.
        indicators : dict[int, Indicator]
            Dictionary mapping block IDs to their corresponding indicators.

        Returns
        -------
        float
            The calculated provision value.
        """
        # Update the DataFrame with the values from the solution variables
        update_df = self._clear_df.add(self._vars_to_df(X, indicators))

        # If the service capacity for the current type is zero, return zero
        if update_df[service_type.name].sum() == 0:
            return 0

        # Calculate the provision using the Provision instance
        gdf = self._provision_instance.calculate(service_type, update_df, self_supply=True)
        return self._provision_instance.total(gdf)

    def recalculate_all(self, X: List[Variable], indicators: Dict[int, Indicator]) -> Dict[str, float]:
        """
        Recalculate provisions for all service types.

        This method recalculates the provision for all available service types and returns the results
        as a dictionary.

        Parameters
        ----------
        X : list[Variable]
            List of variables representing the current solution.
        indicators : dict[int, Indicator]
            Dictionary mapping block IDs to their corresponding indicators.

        Returns
        -------
        dict[str, float]
            A dictionary with service type names as keys and their recalculated provisions as values.
        """
        return {
            st.name: provision
            for st in self._provision_instance.city_model.service_types
            if (provision := self.calculate_provision(X, st, indicators)) is not None
        }
