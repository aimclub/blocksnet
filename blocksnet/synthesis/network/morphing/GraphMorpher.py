from blocksnet.analysis.network.classification.core import NetworkClassifier
from blocksnet.enums import SettlementCategory
from .utils import *
from blocksnet.config.log.config import log_config

from networkx import check_planarity
from loguru import logger
from tqdm import tqdm
import sys
import os

CURRENT_FILE = os.path.basename(__file__)

logger.remove()

logger.add(
    sys.stderr,
    level="INFO",
    filter=lambda record: record["name"] == __name__ or record["module"] == CURRENT_FILE
)

log_config.disable_tqdm = True

class GraphMorpher:
    def __init__(self, graph : nx.MultiDiGraph, target_class: SettlementCategory, classifier: NetworkClassifier, simplify: bool = False):
        if simplify:
            self.orig_graph = simplify_graph(graph)
        self.orig_graph = graph
        self.graph = self.orig_graph.copy()
        self.target_class = target_class
        self.model = classifier
        self.nodes = list(self.orig_graph.nodes)
        self.gdf = graph_to_gdf(self.orig_graph)
        self.dist_matrix = compute_distance_matrix(self.orig_graph, self.gdf)

        self.adj = adj_matrix(self.orig_graph)
        self.orig_adj = adj_matrix(self.orig_graph)

        self.classes = [sc.value for sc in SettlementCategory]
        self.classes_wo_tc = self.classes.copy()
        self.classes_wo_tc.remove(self.target_class.value)

        orig_res = self.model.run([self.orig_graph]).iloc[0]
        self.orig_clustering = orig_res['avg_clustering']
        self.orig_assortativity = orig_res['assortativity']
        self.orig_class = orig_res['category']
        self.avg_edge_length = orig_res['avg_edge_length']

        self.cash = dict()
        self.history = []

        logger.info(f"Original class: {self.orig_class}")

        self.exclude_edges = []

    def _strongest_wrong_class(self, res):
        probas = {cat : res[cat] for cat in self.classes_wo_tc}
        ans = max(probas, key=probas.get)
        return ans
    
    def _to_key(self, G):
        edges = tuple(sorted(G.edges()))
        return hash(edges)

    def _shuffle(self, arr: np.ndarray, seed: int | None = None):
        if seed:
            np.random.seed(seed)
        np.random.shuffle(arr)
        return arr
    
    def _get_res(self):
        key = self._to_key(self.graph)
        if key in self.cash:
            res = self.cash[key]
        else:
            res = self.model.run([self.graph]).iloc[0]
            self.cash[key] = res
        return res

    def _get_candidates_to_add(self, target_proba, n_candidates: int | None = None, seed: int | None = None, len_constraint: bool = False):
        non_edges = np.argwhere(self.adj == 0)
        
        if len(self.exclude_edges) > 0:
            edges_to_remove = np.array([edge for edge, score in self.exclude_edges])
            non_edges_sorted = np.sort(non_edges, axis=1)
            remove_set = {tuple(edge) for edge in edges_to_remove}
            mask = np.array([tuple(edge) not in remove_set for edge in non_edges_sorted])
            non_edges_sorted = non_edges_sorted[mask]

        non_edges = self._shuffle(non_edges, seed=seed)
        to_add_candidates = []
        max_candidates = n_candidates if n_candidates else len(non_edges)
        for u, v in tqdm(non_edges, desc=f"Searching and processing candidates to add"):
            if u == v:
                continue
            if not len_constraint and self.dist_matrix[u][v] > self.avg_edge_length:
                continue

            self.graph.add_edge(self.nodes[u], self.nodes[v])
            if check_planarity(self.graph)[0]:
                curr_res = self._get_res()
                curr_score = self._count_score(curr_res, target_proba)
                if curr_score:
                    to_add_candidates.append(((u, v), curr_score))
            self.graph.remove_edge(self.nodes[u], self.nodes[v])
            if len(to_add_candidates) >= max_candidates:
                break
            
        return to_add_candidates

    def _get_candidates_to_delete(self, target_proba,  seed: int | None = None):
        current_adj = adj_matrix(self.graph)
        existing_edges = np.array(current_adj.nonzero()).T

        if len(self.exclude_edges) > 0:
            edges_to_remove = np.array([edge for edge, score in self.exclude_edges])
            existing_edges_sorted = np.sort(existing_edges, axis=1)
            remove_set = {tuple(edge) for edge in edges_to_remove}
            mask = np.array([tuple(edge) not in remove_set for edge in existing_edges_sorted])
            existing_edges = existing_edges_sorted[mask]

        existing_edges = self._shuffle(existing_edges, seed=seed)
        delete_candidates = [(u, v) for u, v in existing_edges if self.orig_adj[u, v] == 0]
        to_delete_candidates = []
        for u, v in tqdm(delete_candidates, desc=f"Searching and processing candidates to delete"):
            if self.orig_adj[u, v] == 0:
                continue
            if not self.graph.has_edge(self.nodes[u], self.nodes[v]):
                continue

            self.graph.remove_edge(self.nodes[u], self.nodes[v])
            curr_res = self._get_res()
            curr_score = self._count_score(curr_res, target_proba)
            self.graph.add_edge(self.nodes[u], self.nodes[v])
            if curr_score:
                to_delete_candidates.append(((u, v), curr_score))

        return to_delete_candidates
    

    def _get_edge_to_exclude(self):
        if len(self.history) < 4:
            return None
        if self.history[-1] == self.history[-3] and self.history[-2] == self.history[-4]:
            return (self.history[-1][1], 5)
        return None


    def _count_score(self, curr_res, target_proba):
        curr_target = curr_res[self.target_class.value]
        delta_target = curr_target - target_proba
        curr_clastering = curr_res['avg_clustering']
        delta_ratio_clustering = abs((self.orig_clustering - curr_clastering) / self.orig_clustering)
        curr_assortativity = curr_res['assortativity']
        delta_ratio_assortativity = (self.orig_assortativity - curr_assortativity) / abs(self.orig_assortativity)
        score = delta_target

        if delta_ratio_clustering > 0.1 or delta_ratio_assortativity < -0.1:
            return None

        return score

    def morph(self, n_steps : int = 20, n_candidates_to_add: int | None = 50, seed: int | None = None, len_constraint: bool = False):
        for step in range(n_steps):
            res_before = self._get_res()
            target_proba = res_before[self.target_class.value]
            pred = res_before['category']
            if pred == self.target_class:
                logger.success(f"Target class is reached after {step} steps")
                break
            
            # finding candidates
            logger.info(f"step: {step + 1}: Searching and processing candidates")
            if len(self.exclude_edges) > 0:
                self.exclude_edges = [(edge, n - 1) for edge, n in self.exclude_edges if n != 1]
            new_excl = self._get_edge_to_exclude()
            if new_excl is not None:
                self.exclude_edges.append(new_excl)

            to_add_candidates = self._get_candidates_to_add(target_proba, n_candidates=n_candidates_to_add, seed=seed, len_constraint=len_constraint)
            delete_candidates = self._get_candidates_to_delete(target_proba, seed=seed)
            candidates = to_add_candidates + delete_candidates

            if len(candidates) == 0:
                logger.warning("No candidates avaliable. Stopping...")
                logger.info("Try with different settings")
                break

            candidates.sort(key=lambda x: -x[1]) 
            (u, v), score = candidates[0]

            # modify graph
            old_val = self.adj[u][v]
            action = "➕ Add" if old_val == 0 else "➖ Remove"
            if old_val == 0:
                self.graph.add_edge(self.nodes[u], self.nodes[v])
                self.history.append(("add", tuple(sorted((u, v)))))
            else:
                self.graph.remove_edge(self.nodes[u], self.nodes[v])
                self.history.append(("delete", tuple(sorted((u, v)))))

            self.adj = adj_matrix(self.graph)
            res = self._get_res()
            target_proba = res[self.target_class.value]
            logger.info(f"step: {step + 1}: {action} edge ({self.nodes[u]},{self.nodes[v]}) | target proba: {target_proba:.4f}")
        else:
            logger.info("Not enough steps to reach target class")

        return self.orig_graph, self.graph