from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from blocksnet.machine_learning.gnn_model.service import ServiceGNN
from blocksnet.machine_learning.gnn_model.optimization_problem import ServiceOptimizationProblem
from blocksnet.analysis.services import collocation_matrix
from blocksnet.machine_learning.gnn_model.optimization_problem import ServiceOptimizationProblem
from blocksnet.optimization.recommends.pareto_utils import get_recommendations
import geopandas as gpd
import math
import pandas as pd
import torch
import pickle
import osmnx as ox
import os
from torch_geometric.data import Data
torch.serialization.add_safe_globals([Data])

blocks_gdf = gpd.read_parquet(os.path.normpath('D:/Загрузки Яндекс/blocks.parquet'))
capacity_cols = [col for col in blocks_gdf.columns if col.startswith('capacity_')]

def adjust_value(x):
    val = math.ceil(x / 200)
    if val >= 5:
        return min(4, int(math.log(val + 1) * 3))
    return val

collocation_matrix = collocation_matrix(blocks_gdf)

service_columns = [col for col in blocks_gdf.columns if col.startswith('capacity_')]
service_types = [col.replace('capacity_', '') for col in service_columns]

services_list = []
for block_id in blocks_gdf.index:
    for service_col in service_columns:
        capacity = blocks_gdf.loc[block_id, service_col]
        if capacity > 0:
            service_type = service_col.replace('capacity_', '')
            services_list.append({
                'block_id': block_id,
                'service_type': service_type,
                'count': capacity
            })

services_gdf = pd.DataFrame(services_list)
services_gdf = services_gdf.loc[services_gdf.index.repeat(services_gdf['count'])].reset_index(drop=True)
services_gdf = services_gdf.drop(columns=["count"])
services_gdf['service_type'] = services_gdf['service_type'].astype(str)
all_services = services_gdf['service_type'].unique()
service_to_idx = {s: i for i, s in enumerate(all_services)}
distances = pd.read_pickle(os.path.normpath('D:/Загрузки Яндекс/accessibility_matrix.pickle'))
district_gdf = ox.geocode_to_gdf('Санкт-Петербург, Василеостровский район')[['geometry']]

if blocks_gdf.crs != district_gdf.crs:
    blocks_gdf = blocks_gdf.to_crs(district_gdf.crs)

target_gdf = blocks_gdf[blocks_gdf.intersects(district_gdf.geometry.iloc[0])]
target_block_ids = list(set(target_gdf.index))
blocks_gdf['block_id'] = blocks_gdf.index

filtered_blocks = blocks_gdf[blocks_gdf['block_id'].isin(target_block_ids)].reset_index(drop=True)
filtered_services = services_gdf[services_gdf['block_id'].isin(target_block_ids)].copy().reset_index(drop=True)
distance_matrix = distances.loc[target_block_ids, target_block_ids].values if isinstance(distances, pd.DataFrame) else distances[np.ix_(target_block_ids, target_block_ids)]

graph_data = torch.load('blocksnet/machine_learning/gnn_model/model/graph_data.pt', weights_only=False)
if isinstance(graph_data, dict):
    graph_data = Data(
        x=graph_data['x'],
        edge_index=graph_data['edge_index'],
        edge_attr=graph_data['edge_attr']
    )

checkpoint = torch.load(os.path.normpath('blocksnet/machine_learning/gnn_model/model/gnn_model.pth'))

gnn = ServiceGNN(
    input_dim=checkpoint['input_dim'],
    hidden_dim=checkpoint['hidden_dim'],
    output_dim=checkpoint['output_dim']
)

gnn.load_state_dict(checkpoint['state_dict'])
gnn.eval()

with open('C:/Users/Poli/Desktop/mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)
    service_to_idx = mappings['service_to_idx']
    block_to_node = mappings['block_to_node']

problem = ServiceOptimizationProblem(
    blocks=filtered_blocks,         
    services=filtered_services,     
    collocation_matrix=collocation_matrix,
    distance_matrix=distance_matrix,  
    service_to_idx=service_to_idx,
    target_blocks=[4892, 2685, 6624],  
    gnn_model=gnn,                 
    graph_data=graph_data,        
    n_recommendations=3,            
    max_time=15                     
)

def optimization_callback(algorithm):
    print(f"Gen: {algorithm.n_gen}:")
    print(f"  Best F value: {algorithm.pop.get('F').min(axis=0)}")
    print(f"  Number of non-dominated solutions: {len(algorithm.opt)}")
    print("-" * 50)


print("\nStart of optimization...")
print(f"Target blocks: {problem.target_blocks}")
print(f"Number of recommendations per block: {problem.n_recommendations}")

algorithm = NSGA2(
    pop_size=200,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=0.8, eta=20),
    mutation=PM(eta=25, prob=0.1),
    eliminate_duplicates=True
)


res = minimize(
    problem,
    algorithm,
    ('n_gen', 20),
    seed=42,
    callback=optimization_callback,  
    verbose=True
)
print("\nOptimization completed")
print(f"Total gens: {algorithm.n_gen}")
print(f"Total evals: {algorithm.evaluator.n_eval}")
print(f"Found {len(res.X)} optimal solutions")

recommendations = get_recommendations(res, problem)