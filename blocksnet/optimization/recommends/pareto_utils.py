import pandas as pd
import torch
import numpy as np
from typing import List, Dict, Any

def get_recommendations(res: Any, problem: Any) -> List[Dict[str, Any]]:
    """
    Формирование списока рекомендаций на основе решений оптимизации

    Params:
        res -- Результат оптимизации, содержащий решения и метрики 
        problem -- Экземпляр задачи оптимизации, содержащий данные и методы для анализа

    Returns:
        Отсортированный список рекомендаций для каждого решения
    """
    # извлечение решения и метрики из результатов оптимизации
    solutions = res.X  # массив решений
    metrics = res.F    # массив значений целевых функций

    # получение предсказания GNN-модели для всех узлов графа
    with torch.no_grad():
        gnn_scores = problem.gnn_model(problem.graph_data).numpy()

    recommendations = []

    def calculate_normalized_shannon_entropy(services: List[str], 
                                             service_to_idx: Dict[str, int]) -> float:
        """
        Вычисление нормализованного индекса Шеннона для заданного списка сервисов

        Params:
            services -- Список сервисов для расчета индекса
            service_to_idx -- Словарь соответствий названий сервисов их индексам

        Returns:
            Нормализованное значение индекса Шеннона
        """
        if not services:
            return 0.0

        all_services = list(service_to_idx.keys())  
        counts = pd.Series(services).value_counts(normalize=True)  
        full_counts = counts.reindex(all_services, fill_value=0.0)  
        entropy = -np.sum(full_counts * np.log(full_counts + 1e-9))  
        max_entropy = np.log(len(all_services)) 
        return entropy / max_entropy if max_entropy > 0 else 0.0  

    # обработка каждого решения
    for sol, (div, types) in zip(solutions, metrics):
        rec = {
            'diversity': -div,  
            'unique_types': types,  
            'gnn_score': 0.0, 
            'recommendations': {},  
            'diversity_changes': {} 
        }

        total_gnn_score = 0  
        valid_blocks = 0    

        # обработка каждого целевого квартала
        for i, block_id in enumerate(problem.target_blocks):
            node_idx = problem.block_to_node.get(block_id)  
            if node_idx is None:
                continue

            start = i * problem.n_recommendations  
            services = []  

            # добавление рекомендованных сервисов
            for j in range(problem.n_recommendations):
                try:
                    service_idx = int(sol[start + j])  
                    service = problem.idx_to_service[service_idx]  
                    services.append(service)
                    total_gnn_score += gnn_scores[node_idx, service_idx]  
                except (IndexError, KeyError):
                    continue

            # существующие сервисы в квартале
            existing = problem.services[
                problem.services['block_id'] == block_id
            ]['service_type'].tolist()

            # расчет изменений разнообразия
            original_div = calculate_normalized_shannon_entropy(existing, problem.service_to_idx)
            new_div = calculate_normalized_shannon_entropy(existing + services, problem.service_to_idx)

            rec['diversity_changes'][block_id] = {
                'original': original_div,  
                'new': new_div,          
                'delta': new_div - original_div 
            }
            rec['recommendations'][block_id] = services 
            valid_blocks += 1

        # усредненная GNN-оценка по всем валидным кварталам
        rec['gnn_score'] = (
            total_gnn_score / (valid_blocks * problem.n_recommendations)
            if valid_blocks > 0 else 0
        )
        recommendations.append(rec)

    # сортировка рекомендаций по приоритету
    return sorted(recommendations, key=lambda x: (
        -x['diversity'],   
        -x['gnn_score'],   
        x['unique_types']  
    ))