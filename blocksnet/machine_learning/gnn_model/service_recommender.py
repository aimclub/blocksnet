from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

class ServiceRecommender:
  def __init__(self, 
              gnn: Any, 
              collocation_matrix: pd.DataFrame, 
              service_to_idx: Dict[str, int], 
              block_to_node: Dict[int, int], 
              time_matrix: np.ndarray, 
              max_time: int = 15) -> None:
    """
    Инициализия рекомендательной системы для сервисов в городских кварталах

    Params:
        gnn -- Предобученная GNN-модель для оценки сервисов
        collocation_matrix -- Матрица совместимости между сервисами
        service_to_idx -- Словарь соответствий названий сервисов их индексам
        block_to_node -- Отображение ID кварталов в индексы узлов графа
        time_matrix -- Матрица времени перемещения между кварталами
        max_time -- Максимальное время (в минутах) для учета соседей

    Returns:
        None
    """
    self.gnn = gnn
    self.collocation_matrix = collocation_matrix
    self.service_to_idx = service_to_idx
    self.idx_to_service = {v: k for k, v in service_to_idx.items()}
    self.block_to_node = block_to_node
    self.time_matrix = time_matrix
    self.max_time = max_time

    # проверка корректности данных
    if len(block_to_node) != time_matrix.shape[0]:
      self.valid_blocks = set(block_to_node.values()) & set(range(time_matrix.shape[0]))
    else:
      self.valid_blocks = set(block_to_node.values())


  def _calculate_shannon(self, services_list: List[str]) -> float:
    """
    Вычисление индекса Шеннона для заданного списка сервисов

    Params:
        services_list -- Список сервисов для расчета энтропии

    Returns:
        Значение индекса Шеннона
    """
    counts = pd.Series(services_list).value_counts(normalize=True)
    return -np.sum(counts * np.log(counts + 1e-9))


  def _get_nearby_services(self, block_id: int, 
                           graph_data: Any) -> Dict[str, float]:
    """
    Получение взвешенного словаря сервисов, доступных в соседних кварталах

    Params:
        block_id -- id квартала, для которого вычисляются ближайшие сервисы
        graph_data -- Данные графа, содержащие информацию о сервисах

    Returns:
        Словарь, где ключи — названия сервисов, а значения — их веса
    """
    node_idx = self.block_to_node.get(block_id)
    if node_idx is None or node_idx not in self.valid_blocks:
      return {}

    try:
      # поиск соседних кварталов в пределах максимального времени перемещения
      neighbor_indices = np.where(
        (self.time_matrix[node_idx] <= self.max_time) &
        (self.time_matrix[node_idx] > 0)
      )[0]
    except IndexError:
      return {}

    nearby_services = {}
    x = graph_data.x.numpy()

    for neighbor_idx in neighbor_indices:
      if neighbor_idx >= x.shape[0]:
        continue

      try:
        # определение сервисов в соседнем квартале
        services_in_neighbor = np.where(
            x[neighbor_idx, :len(self.service_to_idx)] > 0.5
        )[0]
        time = self.time_matrix[node_idx, neighbor_idx]
        weight = np.exp(-time / 5)

        for idx in services_in_neighbor:
          if idx in self.idx_to_service:
            service = self.idx_to_service[idx]
            nearby_services[service] = nearby_services.get(service, 0) + weight
      except IndexError:
        continue

    return nearby_services


  def recommend_for_block(self, block_id: int, 
                          graph_data: Any, 
                          top_n: int = 3) -> List[str]:
    """
    Рекомендация сервисов для заданного квартала

    Params:
        block_id -- id целевого квартала
        graph_data -- Данные графа, содержащие информацию о сервисах
        top_n -- Количество рекомендуемых сервисов

    Returns:
        Список названий рекомендуемых сервисов
    """
    node_index = self.block_to_node.get(block_id)
    if node_index is None or node_index >= graph_data.x.shape[0]:
      raise ValueError(f"Block {block_id} not found in graph")

    x = graph_data.x.numpy()
    # текущие сервисы в квартале
    current_services = [
            self.idx_to_service[idx] for idx in np.where(x[node_index, :len(self.service_to_idx)] > 0.5)[0]
            if idx in self.idx_to_service
    ]
    # ближайшие сервисы в соседних кварталах
    nearby_services = self._get_nearby_services(block_id, graph_data)
    current_diversity = self._calculate_shannon(current_services)
    scores = {}

    for service in self.collocation_matrix.index:
        if service in current_services:
          continue

        # проверка совместимости нового сервиса с существующими
        incompatible = False
        for existing in current_services + list(nearby_services.keys()):
          try:
            if self.collocation_matrix.loc[service, existing] <= 0.00001:
              incompatible = True
              break
          except KeyError:
            continue

        if incompatible:
          continue

        # расчет прироста разнообразия
        new_diversity = self._calculate_shannon(current_services + [service])
        diversity_gain = new_diversity - current_diversity

        # расчет средней совместимости
        total_compat = 0
        total_weight = 0

        for existing in current_services:
          try:
            total_compat += self.collocation_matrix.loc[service, existing]
            total_weight += 1
          except KeyError:
            continue

        for neighbor, weight in nearby_services.items():
          try:
            total_compat += self.collocation_matrix.loc[service, neighbor] * weight
            total_weight += weight
          except KeyError:
            continue

        avg_compat = total_compat / total_weight if total_weight > 0 else 0

        # расчет плотности сервисов в соседних кварталах
        density = sum(
          self.collocation_matrix.loc[service, neighbor] * weight
          for neighbor, weight in nearby_services.items()
          if neighbor in self.collocation_matrix.index
        ) / sum(nearby_services.values()) if nearby_services else 0

        # комбинированная оценка сервиса
        scores[service] = 0.3 * avg_compat + 0.5 * diversity_gain + 0.2 * density

    # сортировка сервисов по убыванию оценки
    sorted_services = sorted(scores.items(), key=lambda x: -x[1])
    return [s[0] for s in sorted_services[:top_n]]