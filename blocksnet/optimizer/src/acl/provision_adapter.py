from typing import Dict, List

import pandas as pd

from blocksnet import LandUse, ServiceType
from blocksnet.method.annealing_optimizer import Indicator, Variable
from blocksnet.method.provision import Provision
from blocksnet.models import City


class ProvisionAdapter:
    """
    Adapter class for interacting with the provision calculations.

    This class provides methods to calculate provisions based on
    city blocks, their land use, and service indicators.
    """

    def __init__(self, city_model: City, blocks_lu: Dict[int, LandUse], blocks_gsi: Dict[int, float]):
        """
        Initializes the ProvisionAdapter with city model and block data.

        Parameters
        ----------
        city_model : City
            The city model containing block and service type information.
        blocks_lu : Dict[int, LandUse]
            Dictionary mapping block IDs to their respective land uses.
        blocks_gsi : Dict[int, float]
            Dictionary mapping block IDs to their green space index (GSI) values.
        """
        self.provision_instance: Provision = Provision(city_model=city_model, verbose=False)
        self.clear_df: pd.DataFrame = self._get_clear_df(blocks_lu.keys())

    def _vars_to_df(self, X: List[Variable], indicators: Dict[int, Indicator]) -> pd.DataFrame:
        """
        Converts the list of variables and indicators into a DataFrame.

        Parameters
        ----------
        X : list[Variable]
            List of variables representing the current solution.
        indicators : dict[int, Indicator]
            Dictionary mapping block IDs to their corresponding indicators.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing data about the blocks and their service capacities.
        """
        service_types = {x.service_type for x in X}
        df = pd.DataFrame(
            [
                {
                    "block_id": x.block.id,
                    "population": indicators[x.block.id].population,
                    x.service_type.name: x.capacity,
                }
                for x in X
            ]
        )
        return df.groupby("block_id").agg({"population": "min", **{st.name: "sum" for st in service_types}})

    def _get_clear_df(self, blocks: List[int]) -> pd.DataFrame:
        """
        Constructs a DataFrame for provision assessment so the blocks being changed are treated as cleared.

        Parameters
        ----------
        blocks : list[int]
            List of changing block IDs.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing information related to blocks being changed for provision assessment.
        """
        gdf = self.provision_instance.city_model.get_blocks_gdf()
        gdf = gdf[gdf.index.isin(blocks)]
        df = gdf[["population"]].copy()
        df["population"] = -df["population"]
        df.sum()
        for column in [column for column in gdf.columns if "capacity_" in column]:
            st_name = column.removeprefix("capacity_")
            df[st_name] = -gdf[column]
        return df
    
    def calculate_provision(self, X: List[Variable], service_type: ServiceType, indicators: Dict[int, Indicator]) -> float:
        """
        Calculate the provision for a specific service type.

        Parameters
        ----------
        X : List[Variable]
            List of variables representing the current solution.
        service_type : ServiceType
            The service type to calculate provision for.
        indicators : Dict[int, Indicator]
            Dictionary mapping block IDs to their corresponding indicators.

        Returns
        -------
        float
            The calculated provision value.
        """
        pass  # TODO: Implement provision calculation logic
