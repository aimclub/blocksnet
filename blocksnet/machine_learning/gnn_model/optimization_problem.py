from collections import defaultdict
from typing import Dict, List, Tuple, Any
from pymoo.core.problem import Problem
import numpy as np
import pandas as pd
import torch

class ServiceOptimizationProblem(Problem):
    def __init__(self, 
                 blocks: pd.DataFrame, 
                 services: pd.DataFrame, 
                 collocation_matrix: pd.DataFrame, 
                 distance_matrix: np.ndarray,
                 service_to_idx: Dict[str, int], 
                 target_blocks: List[int], 
                 gnn_model: torch.nn.Module, 
                 graph_data: Any,
                 n_recommendations: int = 3, 
                 max_time: int = 15, 
                 gnn_weight: float = 0.7) -> None:
        """
        Функция для инициализации задачи оптимизации для рекомендации сервисов в городских кварталах

        Params:
            blocks -- GeoDataFrame, содержащий информацию о городских кварталах
            services -- GeoDataFrame, содержащий существующие сервисы в кварталах
            collocation_matrix -- Матрица совместимости между сервисами
            distance_matrix -- Матрица времени/расстояний между кварталами
            service_to_idx -- Словарь соответствий названий сервисов их индексам
            target_blocks -- Список ID целевых кварталов, где нужны рекомендации
            gnn_model -- Предобученная GNN-модель для оценки рекомендаций
            graph_data -- Структура данных графа, используемая GNN-моделью
            n_recommendations -- Количество рекомендаций на каждый целевой квартаx
            max_time -- Максимальное время (в минутах) для учета соседей
            gnn_weight -- Вес оценки GNN в целевой функции

        Returns:
            None
        """
        self.blocks = blocks.reset_index(drop=True)
        self.services = services
        self.collocation_matrix = collocation_matrix
        self.distance_matrix = distance_matrix
        self.service_to_idx = service_to_idx
        self.idx_to_service = {v: k for k, v in service_to_idx.items()}
        self.target_blocks = target_blocks
        self.n_recommendations = n_recommendations
        self.max_time = max_time
        self.gnn_model = gnn_model
        self.graph_data = graph_data
        self.gnn_weight = gnn_weight

        # подготовка данных
        self._prepare_graph_structures()
        self.all_services = list(service_to_idx.keys())

        # инициализация проблемы для pymoo
        n_var = len(target_blocks) * n_recommendations
        super().__init__(n_var=n_var, n_obj=2, xl=0, xu=len(service_to_idx)-1, vtype=int)


    def _prepare_graph_structures(self) -> None:
        """
        Метод для подготовки вспомогательных структур данных для эффективного доступа во время оптимизации
        Он создает отображения между кварталами и узлами графа, а также определяет соседей
        в пределах максимального времени перемещения для каждого квартала

        Returns:
            None
        """
        # отображение id кварталов в индексы узлов
        self.block_to_node = {block_id: idx for idx, block_id in enumerate(self.blocks['block_id'])}
        self.node_to_block = {v: k for k, v in self.block_to_node.items()}

        # нахождение соседей для каждого квартала в пределах максимального времени перемещения
        self.neighbors = {}
        for i, block_id in enumerate(self.blocks['block_id']):
            mask = self.distance_matrix[i] <= self.max_time
            self.neighbors[block_id] = list(np.array(self.blocks['block_id'])[mask])


    def _get_nearby_services(self, block_id: int) -> Dict[str, float]:
        """
        Метод для получения взвешенного словаря сервисов, доступных в соседних кварталах

        Params:
            block_id -- id квартала, для которого вычисляются ближайшие сервисы

        Returns:
            Словарь, где ключи — названия сервисов, а значения — их веса
        """
        nearby = defaultdict(float)
        for neighbor_id in self.neighbors.get(block_id, []):
            if neighbor_id not in self.block_to_node:
                continue

            time = self.distance_matrix[
                self.block_to_node[block_id],
                self.block_to_node[neighbor_id]
            ]
            if time <= 0 or neighbor_id == block_id:
                continue

            # учет существующих сервисов и рекомендаций
            neighbor_services = self.services[
                self.services['block_id'] == neighbor_id
            ]['service_type'].tolist()

            weight = 1.0 / (1 + time)
            for service in neighbor_services:
                nearby[service] += weight

        return nearby


    def calculate_entropy(self, services: List[str]) -> float:
        """
        Функция для вычисления нормализованного индекса Шеннона для заданного списка сервисов

        Params:
            services -- Список сервисов для расчета энтропии

        Returns:
            Значение нормализованной энтропии
        """
        if not services:
            return 0.0

        counts = pd.Series(services).value_counts(normalize=True)
        full_counts = counts.reindex(self.all_services, fill_value=0.0)
        entropy = -np.sum(full_counts * np.log(full_counts + 1e-9))
        max_entropy = np.log(len(self.all_services))
        return entropy / max_entropy if max_entropy > 0 else 0.0


    def _evaluate(self, 
                  X: np.ndarray, 
                  out: Dict[str, Any], *args, **kwargs) -> None:
        """
        Основная функция оценки для оптимизации

        ПаParams:
            X -- Массив решений, где каждая строка представляет одно решение
            out -- Словарь выходных данных, куда записываются результаты оценки

        Returns:
            None
        """
        n_solutions = X.shape[0]
        f1 = np.zeros(n_solutions)  # основная метрика (минимизируется)
        f2 = np.zeros(n_solutions)  # количество уникальных сервисов (максимизируется)

        # получение предсказания GNN
        with torch.no_grad():
            self.gnn_model.eval()
            gnn_scores = self.gnn_model(self.graph_data).numpy()

        # нормализация оценки GNN
        gnn_min = gnn_scores.min(axis=0, keepdims=True)
        gnn_max = gnn_scores.max(axis=0, keepdims=True)
        gnn_norm = (gnn_scores - gnn_min) / (gnn_max - gnn_min + 1e-9)

        for sol_idx in range(n_solutions):
            solution = X[sol_idx, :]
            total_score = 0.0
            total_penalty = 0.0
            unique_services = set()

            # сбор всех рекомендации
            block_recommendations = {}
            for block_idx, block_id in enumerate(self.target_blocks):
                start = block_idx * self.n_recommendations
                services = [
                    self.idx_to_service[int(solution[start + j])]
                    for j in range(self.n_recommendations)
                    if int(solution[start + j]) in self.idx_to_service
                ]
                block_recommendations[block_id] = services
                unique_services.update(services)

            # оценка каждой рекомендации
            for block_idx, block_id in enumerate(self.target_blocks):
                node_idx = self.block_to_node.get(block_id)
                if node_idx is None:
                    continue

                existing = self.services[
                    self.services['block_id'] == block_id
                ]['service_type'].tolist()
                rec_services = block_recommendations.get(block_id, [])

                # штрафы за невалидные рекомендации
                penalty = 0.0
                # штраф за дублирование существующих сервисов
                penalty += sum(2.0 for s in rec_services if s in existing)
                # штраф за повторение сервисов в рекомендациях
                penalty += sum(1.0 for s in set(rec_services)
                          if rec_services.count(s) > 1)
                total_penalty += penalty

                # расчет метрик
                original_entropy = self.calculate_entropy(existing)
                new_entropy = self.calculate_entropy(existing + rec_services)
                entropy_gain = new_entropy - original_entropy

                # совместимость с существующими сервисами
                compat_score = np.mean([
                    self.collocation_matrix.loc[s, e]
                    for s in rec_services
                    for e in existing
                    if s in self.collocation_matrix.index and
                       e in self.collocation_matrix.index
                ]) if existing else 0.0

                # потность в соседних кварталах
                nearby = self._get_nearby_services(block_id)
                density_score = sum(
                    self.collocation_matrix.loc[s, n] * w
                    for s in rec_services
                    for n, w in nearby.items()
                    if s in self.collocation_matrix.index and
                       n in self.collocation_matrix.index
                ) / (sum(nearby.values()) + 1e-9) if nearby else 0.0

                # оценка GNN
                gnn_score = np.mean([
                    gnn_norm[node_idx, self.service_to_idx[s]]
                    for s in rec_services
                    if s in self.service_to_idx
                ]) if rec_services else 0.0

                # итоговая оценка
                total_score += (
                    0.3 * compat_score +
                    0.4 * entropy_gain +
                    0.2 * density_score +
                    0.1 * gnn_score
                )

            # финальные метрики
            f1[sol_idx] = -total_score + total_penalty  # минимизация
            f2[sol_idx] = len(unique_services)         # максимизация

        out["F"] = np.column_stack([f1, f2])