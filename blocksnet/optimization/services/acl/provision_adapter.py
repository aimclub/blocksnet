from typing import Dict, List, Set
from ....analysis.provision import competitive_provision, provision_strong_total
from ....config import service_types_config
from ....relations import get_accessibility_context
import pandas as pd


class ProvisionAdapter: 
    def __init__(self, blocks_lus, accessibility_matrix,  blocks_df: pd.DataFrame, services_containers):
        # Init provisions_dfs
        self.provisions_dfs = self._initialize_provisions_dfs(list(blocks_lus), accessibility_matrix, blocks_df, services_containers)
        self.provisions_dfs = {
            service_type: provision_df
            for service_type, provision_df in self.provisions_dfs.items()
            if provision_df.demand.sum() > 0
        }

    def _initialize_provisions_dfs(self, blocks_ids: list[int], 
                                   accessibility_matrix, 
                                   blocks_df, 
                                   services_containers) -> dict[str, pd.DataFrame]:
        acc_mx = accessibility_matrix
        service_types = self.service_types

        provisions_dfs = {}

        for service_type in service_types:
            services_df = services_containers[service_type].services_df.copy()
            services_df.loc[blocks_ids, "capacity"] = 0
            _, demand, accessibility = service_types_config[service_type].values()

            provision_df, _ = competitive_provision(blocks_df.join(services_df), acc_mx, demand, accessibility)

            context_acc_mx = get_accessibility_context(acc_mx, provision_df.loc[blocks_ids], accessibility, out=False)
            provisions_dfs[service_type] = provision_df.loc[context_acc_mx.index]
        return provisions_dfs


    def calculate_provision(
        self, X: List[int], st: ServiceType | None, variables_df: DataFrame, accessibility_matrix 
    ) -> float:
        if st is not None:

            delta_df = variables_df.groupby("block_id").agg({"total_capacity": "sum"})

            _, demand, accessibility = service_types_config[st.name].values()
            old_provision_df = self.provisions_dfs[st.name]
            old_provision_df.loc[delta_df.index, "capacity"] += delta_df["total_capacity"]
            new_provision_df, _ = competitive_provision(
                old_provision_df, accessibility_matrix, accessibility, demand
            )
            self.provisions_dfs[st.name] = new_provision_df

        return sum(
            [
                provision_strong_total(provision_df) * self.service_types[service_type]
                for service_type, provision_df in self.provisions_dfs.items()
            ]
        )
