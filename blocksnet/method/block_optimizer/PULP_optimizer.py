from blocksnet.method.accessibility.accessibility import Accessibility
from blocksnet.method.base_method import BaseMethod
from blocksnet.method.block_optimizer.lanuse_coefs import LANUSE_INDICATORS
from blocksnet.models.land_use import LandUse
from blocksnet.method.provision.provision import Provision
from blocksnet.models import Block
import pandas as pd
import numpy as np
from typing import  Tuple
from pulp import *
from blocksnet.method.provision.provision import Provision
import matplotlib.pyplot as plt


class Pulp_Optimizer(BaseMethod):
    NEW_LANDUSE: LandUse = None
    ORIG_LANDUSE: LandUse = None
    SELECTED_BLOCK: int = 0
    FREE_AREA: Tuple = (0, 0)
    NEW_SERVICE_TYPES: list = []
    ORIG_SERVICE_TYPES: list = []

    def calculate_free_area(self, selected_block):
        lanuse_coef = 0.8
        return selected_block.site_area * lanuse_coef

    def get_capacity_demand(self):
        gdf = self.city_model.get_blocks_gdf(False)

        selected_block_gdf = gdf.loc[self.SELECTED_BLOCK].to_dict()
        gdf.loc[self.SELECTED_BLOCK, 'population'] = 0

        selected_block = self.city_model[self.SELECTED_BLOCK]
        acc_gdf = Accessibility(
            city_model=self.city_model).calculate(selected_block)

        constants = {}
        min_capacity_left = float('inf')
        min_service = None

        for serv in  list(set(self.NEW_SERVICE_TYPES) | set(self.ORIG_SERVICE_TYPES)):
            idxs = acc_gdf[(acc_gdf.accessibility_to <= serv.accessibility) | (acc_gdf.accessibility_from <= serv.accessibility)]['id']
            capacity_column = f"capacity_{serv.name}"
            context_gdf = gdf.loc[idxs]
            
            demand_local = context_gdf["population"].apply(serv.calculate_in_need).sum()
            capacity_local = context_gdf[capacity_column].sum() - selected_block_gdf[capacity_column]

            demand_global = gdf["population"].apply(serv.calculate_in_need).sum()
            capacity_global = gdf[capacity_column].sum() - selected_block_gdf[capacity_column]

            capacity_left = capacity_global - demand_global
            if capacity_left < min_capacity_left:
                min_capacity_left = capacity_left
                min_service = serv

            constants[serv.name] = {
                'demand_local': demand_local,
                'capacity_local': capacity_local
            }
            
        min_population = max(min_capacity_left / min_service.demand * 1000, 0)

        return constants, min_population

    def get_bricks_df(self):
        bricks_dict_list = []

        for serv in self.NEW_SERVICE_TYPES:
            for brick in serv.bricks:
                brick_dict = {
                    'service_type': serv.name,
                    'capacity': brick.capacity,
                    'area': brick.area,
                    'is_integrated': brick.is_integrated,
                }
                bricks_dict_list.append(brick_dict)
        bricks_df = pd.DataFrame(bricks_dict_list)
        return bricks_df

    def get_optimal_update_df(self, prob, bricks_df):
        n_bricks = len(prob.variables())
        counts = np.zeros((n_bricks))

        service_counts = prob.variables()
        
        for var_idx in range(n_bricks - 1):
            var = service_counts[var_idx]
            service_count = var.value()
            idx = int(var.name.rsplit('_')[-1])
            counts[idx] = service_count
            bricks_df.loc[idx, 'capacity'] = bricks_df.loc[idx, 'capacity'] * service_count
            bricks_df.loc[idx, 'area'] = bricks_df.loc[idx, 'area'] * service_count

        bricks_to_build = np.where(counts != 0)[0]
        bricks_to_build_df = bricks_df.loc[bricks_to_build]

        service_capacities = bricks_to_build_df.groupby(
            'service_type').sum()['capacity'].to_dict()
        
        population = service_counts[-1].value()
        if population > 0:
           service_capacities['population'] = population

        update = {
            self.SELECTED_BLOCK: service_capacities
        }

        update_df = pd.DataFrame.from_dict(update, orient='index')
        return update_df, bricks_to_build_df

    def generate_lp_problem(self, constants, min_population, bricks_df):
        prob = LpProblem("ProvisionOpt", LpMaximize)

        lp_sum_components = []

        service_weight = 1 / len(self.NEW_SERVICE_TYPES)

        service_counts = LpVariable.dicts("", list(bricks_df.index), 0, None, cat=LpInteger)
        population = LpVariable("population", 0, None, cat=LpInteger)

        for serv in self.NEW_SERVICE_TYPES:
            service_bricks_idxs = list(bricks_df[bricks_df.service_type == serv.name].index)

            demand_local, capacity_local = constants[serv.name]['demand_local'], constants[serv.name]['capacity_local']

            fit_function = lpSum(service_counts[n] * bricks_df.loc[n].capacity for n in service_bricks_idxs)

            if demand_local > 0:
                lp_sum_components.append(service_weight * (((fit_function  + capacity_local - (population * serv.demand) / 1000) / demand_local)))
                 
        # Пытаемся добавить насление, если осталось незаполненные места
        prob += population == min_population

        block_area = self.city_model[self.SELECTED_BLOCK].site_area

        indicators = LANUSE_INDICATORS[self.NEW_LANDUSE]

        FSI_max, FSI_min = indicators.FSI_max, indicators.FSI_min

        prob += sum(service_counts[n] * bricks_df.loc[n].area / block_area for n in list(bricks_df.index))  <= FSI_max
        prob += sum(service_counts[n] * bricks_df.loc[n].area / block_area for n in list(bricks_df.index))  >= FSI_min

        prob += sum(i for i in lp_sum_components)

        return prob

    def get_deleting_update_df(self, selected_block_id, delete_population: bool = True):

        service_capacities = {}
        selected_block = self.city_model[selected_block_id]
        for serv in selected_block.all_services:
            service_dict = serv.to_dict()
            service_capacities[service_dict['service_type']] = -service_dict['capacity']

        if delete_population:
            service_capacities['population'] = -selected_block.population

        update = {
            selected_block_id: service_capacities
        }

        update_df = pd.DataFrame.from_dict(update, orient='index')
        return update_df

    def plot(self, optimal_update_df, bricks_to_build_df, total_before, total_after, categories, previous_values, current_values):
        differences = np.round(np.array(current_values) -
                               np.array(previous_values), 2)

        filtered_categories = []
        filtered_differences = []
        filtered_previous_values = []

        highlight_columns = list(optimal_update_df.columns)

        for category, diff, prev in zip(categories, differences, previous_values):
            if diff != 0:
                filtered_categories.append(category)
                filtered_differences.append(diff)
                filtered_previous_values.append(prev)

        previous_values = np.array(filtered_previous_values)
        categories = np.array(filtered_categories)
        differences = np.array(filtered_differences)

        bar_width = 0.5 
        num_bars = len(filtered_categories)

        fig_width = num_bars * bar_width * 1.5

        fig_width = fig_width if fig_width > 10 else 10  
        fig_height = 6  

        if len(categories) == 0:
            plt.text(0.5, 0.5, 'Изменений нет', horizontalalignment='center',
                     verticalalignment='center', fontsize=14, color='red')
            plt.axis('off') 
        else:
            become_higher = differences > 0
            become_lower = differences < 0

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            bars_previous = ax.bar(categories, previous_values,
                                   width=bar_width, label='Previous Values', color='blue')
            
            bars_difference_higher = ax.bar(categories[become_higher], differences[become_higher], width=bar_width,
                                     bottom=previous_values[become_higher], color='green', label='Provision become higher')

            for bar, diff, prev, category in zip(bars_difference_higher, differences[become_higher], previous_values[become_higher], categories[become_higher]):
                yval = bar.get_height() if diff > 0 else 0
                text = ax.text(bar.get_x() + bar.get_width()/2, prev +
                               yval + 0.01, f'{round(diff, 2)}',  ha='center')

            bars_difference_lower = ax.bar(categories[become_lower], differences[become_lower], width=bar_width,
                                     bottom=previous_values[become_lower], color='red', label='Provision become lower')

            for bar, diff, prev, category in zip(bars_difference_lower, differences[become_lower], previous_values[become_lower], categories[become_lower]):
                yval = bar.get_height() if diff > 0 else 0
                text = ax.text(bar.get_x() + bar.get_width()/2, prev +
                               yval + 0.01, f'{round(diff, 2)}',  ha='center')

            ax.set_ylim(0.0, 1.1)

            lanuse_names = f"Changing {self.ORIG_LANDUSE.name.capitalize()} to {self.NEW_LANDUSE.name.capitalize()}"
            total_provision_difference = f"Total provision difference: {round(total_after - total_before, 3)}"

            ax.set_xlabel('City Services\n The services highlighted in orange have just been built')
            ax.set_ylabel('Provision')
            ax.set_title(
                f'Differences of Provisions in city services\n {lanuse_names} \n {total_provision_difference}')
            ax.legend()

            plt.xticks(rotation=45)

            for i, tick in enumerate(plt.gca().get_xticklabels()):
                if tick.get_text() in highlight_columns:
                    tick.set_color('orange')

            plt.tight_layout()

            plt.show()

    def get_provision_diff(self, optimal_update_df):
        prov = Provision(city_model=self.city_model)

        orig_landuse_services = self.city_model.get_land_use_service_types(
            self.ORIG_LANDUSE)
        new_landuse_services = self.city_model.get_land_use_service_types(
            self.NEW_LANDUSE)

        orig_landuse_services_set = set(
            [x.name for x in orig_landuse_services])
        new_landuse_services_set = set([x.name for x in new_landuse_services])

        combined_landuse_services = list(
            orig_landuse_services_set | new_landuse_services_set)

        scenario = {elem: 1/len(combined_landuse_services)
                    for elem in combined_landuse_services}

        gdf, total_before = prov.calculate_scenario(scenario, self_supply=True)
        provision_before = [prov.total_provision(
            gdf[service]) for service in scenario]

        deleting_update_df = self.get_deleting_update_df(
            selected_block_id=self.SELECTED_BLOCK, delete_population=True)

        update_df = optimal_update_df.combine_first(deleting_update_df)

        gdf, total_after = prov.calculate_scenario(
            scenario, update_df=update_df, self_supply=True)
        
        provision_after = [prov.total_provision(
            gdf[service]) for service in scenario]

        return total_before, total_after, combined_landuse_services, provision_before, provision_after

    def calculate(self, selected_block: Block, new_landuse: LandUse):
        self.SELECTED_BLOCK = selected_block.id
        self.NEW_LANDUSE = new_landuse
        self.ORIG_LANDUSE = selected_block.land_use

        self.NEW_SERVICE_TYPES = self.city_model.get_land_use_service_types(new_landuse)
        self.ORIG_SERVICE_TYPES = self.city_model.get_land_use_service_types(selected_block.land_use)

        self.FREE_AREA = self.calculate_free_area(selected_block)
        constants, min_population = self.get_capacity_demand()

        bricks_df = self.get_bricks_df()

        # method
        prob = self.generate_lp_problem(constants, min_population, bricks_df)
        prob.solve(PULP_CBC_CMD(msg=False))

        optimal_update_df, bricks_to_build_df = self.get_optimal_update_df(prob, bricks_df)

        total_before, total_after,categories,previous_values,current_values = self.get_provision_diff(optimal_update_df)

        deleting_update_df = self.get_deleting_update_df(selected_block_id=self.SELECTED_BLOCK, delete_population=True)
        
        return {
            'optimal_update_df': optimal_update_df,
            'bricks_to_build_df': bricks_to_build_df,
            'total_before': total_before,
            'total_after': total_after,
            'categories': categories,
            'previous_values': previous_values,
            'current_values': current_values
        }, deleting_update_df
        
