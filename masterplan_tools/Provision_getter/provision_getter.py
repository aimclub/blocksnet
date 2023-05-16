import pandas as pd
import numpy as np
import psycopg2 as pg
import geopandas as gpd
import networkx as nx
from tqdm.auto import tqdm # pylint: disable=import-error

tqdm.pandas()


class ProvisionModel:
    """
    TODO: add docstring
    TODO: manage UserWarning: pandas only supports SQLAlchemy connectable
    """

    standard_dict = {"kindergartens":61,"schools":120,"universities":13,"hospitals":9,"policlinics":27,"theaters":5,"cinemas":10,
    "cafes":72,"bakeries":72, "fastfoods":72, "music_school":8,"sportgrounds":15,"swimming_pools":50,"conveniences":90,
    "recreational_areas":5000,"pharmacies":50,"playgrounds":550, "supermarkets":992}

    def __init__(
        self,
        blocks: gpd.GeoDataFrame,
        service_name: str = "schools",
        city_crs: int = 32636,
        g: int = 32636,

    ):
        self.blocks = blocks
        self.service_name = service_name
        self.city_crs = city_crs
        self.standard = self.standard_dict[self.service_name],
        self.g = g

    def get_stats(self):
        """
        TODO: add docstring
        """
        g = self.g.copy()
        if not g:
            return 0

        blocks_service = 0
        blocks_bad = 0
        blocks_living = 0
        total = 0
        for key in g:
            total += 1
            try:
                if g.nodes[key]["is_living"] == True:
                    blocks_living += 1
                elif g.nodes[key][f"is_{self.service_name}_service"] == 1:
                    blocks_service += 1
            except KeyError:
                blocks_bad += 1

        print(f"количество кварталов c сервисом {self.service_name}: {blocks_service}")
        print(f"количество жилых кварталов: {blocks_living}")
        print(f"количество кварталов всего: {total}")
        print(f"количество кварталов c ошибкой: {blocks_bad}")

    def get_provision(self):
        g = self.g
        standard = self.standard


        for node in g.nodes:
            if g.nodes[node][f'is_{self.service_name}_service'] == 1:
                neighbors = list(g.neighbors(node))
                capacity = g.nodes[node][f'{self.service_name}_capacity'] 
                if g.nodes[node]['is_living'] == True and g.nodes[node]['population'] > 0 and g.nodes[node][f'provision_{self.service_name}'] < 100:

                    if  g.nodes[node][f'provision_{self.service_name}'] == 0:
                        load = g.nodes[node][f'population_unprov_{self.service_name}'] / 1000 * standard
                        
                    elif  g.nodes[node][f'provision_{self.service_name}'] > 0:
                        load = (g.nodes[node][f'population_unprov_{self.service_name}'] / 1000) * standard

                    if  load <= capacity: 
                          capacity -= load
                          g.nodes[node][f'provision_{self.service_name}'] = 100
                          g.nodes[node][f'id_{self.service_name}'] = node
                          g.nodes[node][f'population_prov_{self.service_name}'] += g.nodes[node][f'population_unprov_{self.service_name}']
                          g.nodes[node][f'population_unprov_{self.service_name}'] -= g.nodes[node][f'population_unprov_{self.service_name}']
                          
                    else:
                          if  capacity > 0:
                              prov_people = (capacity * 1000) / standard
                              capacity -= capacity
                              
                              g.nodes[node][f'id_{self.service_name}'] = node
                              g.nodes[node][f'population_prov_{self.service_name}'] += prov_people
                              g.nodes[node][f'population_unprov_{self.service_name}'] = g.nodes[node][f'population_unprov_{self.service_name}'] - prov_people
                              g.nodes[node][f'id_{self.service_name}'] = node
                              g.nodes[node][f'provision_{self.service_name}'] = (prov_people  * 100) / g.nodes[node]['population']   

                for neighbor in neighbors:
                    if g.nodes[neighbor]['is_living'] == True and g.nodes[neighbor]['population'] > 0 \
                        and g.nodes[neighbor][f'is_{self.service_name}_service'] == 0 and capacity > 0:

                        if g.nodes[neighbor]['is_living'] == True and g.nodes[neighbor]['population'] > 0 and g.nodes[neighbor][f'provision_{self.service_name}'] < 100:

                            if  g.nodes[neighbor][f'provision_{self.service_name}'] == 0:
                                load = (g.nodes[neighbor][f'population_unprov_{self.service_name}'] / 1000) * standard

                            elif  g.nodes[neighbor][f'provision_{self.service_name}'] > 0:
                                load = (g.nodes[neighbor][f'population_unprov_{self.service_name}'] / 1000) * standard

                            if  load <= capacity: 
                                capacity -= load
                                g.nodes[neighbor][f'provision_{self.service_name}'] = 100
                                g.nodes[neighbor][f'id_{self.service_name}'] = node
                                g.nodes[neighbor][f'population_prov_{self.service_name}'] += g.nodes[neighbor][f'population_unprov_{self.service_name}']
                                g.nodes[neighbor][f'population_unprov_{self.service_name}'] -= g.nodes[neighbor][f'population_unprov_{self.service_name}']
                                
                            else:
                                if  capacity > 0:
                                    prov_people = (capacity * 1000) / standard
                                    capacity -= capacity
                                    
                                    g.nodes[neighbor][f'id_{self.service_name}'] = neighbor
                                    g.nodes[neighbor][f'population_prov_{self.service_name}'] += prov_people
                                    g.nodes[neighbor][f'population_unprov_{self.service_name}'] = g.nodes[neighbor][f'population_unprov_{self.service_name}'] - prov_people
                                    g.nodes[neighbor][f'id_{self.service_name}'] = node
                                    g.nodes[neighbor][f'provision_{self.service_name}'] = (prov_people  * 100) / g.nodes[neighbor]['population'] 
                    

        self.g = g
    
    def get_geo(self):
        g = self.g.copy()
        blocks=self.blocks.copy()
        blocks[f'provision_{self.service_name}'] = 0
        blocks[f'id_{self.service_name}'] = 0
        blocks[f'population_prov_{self.service_name}'] = 0 
        blocks[f'population_unprov_{self.service_name}'] = 0
        blocks[f'provision_{self.service_name}'] = 0
        blocks['population'] = 0
  

        for n in g:
            indx = blocks[blocks.index == n].index[0]
            if g.nodes[n]['is_living'] == True:
                if g.nodes[n].get(f'id_{self.service_name}') is not None:
                    blocks.loc[indx, f'id_{self.service_name}'] = g.nodes[n][f'id_{self.service_name}']
                    blocks.loc[indx, f'population_prov_{self.service_name}'] = g.nodes[n][f'population_prov_{self.service_name}']
                    blocks.loc[indx, f'population_unprov_{self.service_name}'] = g.nodes[n][f'population_unprov_{self.service_name}']
                    blocks.loc[indx, f'provision_{self.service_name}'] = g.nodes[n][f'provision_{self.service_name}']
                    blocks.loc[indx, 'population'] = g.nodes[n]['population']
                    
                else:
                    blocks[f'population_unprov_{self.service_name}'][indx] = g.nodes[n][f'population_unprov_{self.service_name}']

        blocks[f'id_{self.service_name}'] = blocks[f'id_{self.service_name}'].astype(int)
        blocks[f'population_prov_{self.service_name}'] =  blocks[f'population_prov_{self.service_name}'].astype(int)
        blocks[f'population_unprov_{self.service_name}'] = blocks[f'population_unprov_{self.service_name}'].astype(int)
        blocks[f'provision_{self.service_name}'] = blocks[f'provision_{self.service_name}'].astype(int)
        blocks['population'] = blocks['population'].astype(int)

        return blocks

    def run(self):
        self.get_stats()
        self.get_provision()
        return self.get_geo()
