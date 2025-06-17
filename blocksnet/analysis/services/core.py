# Импорты библиотек
import pandas as pd
import numpy as np
from itertools import combinations
from loguru import logger
from .schemas import BlocksGdfSchema
from blocksnet.utils.validation import df_schema

CO_OCCURRENCE_COLUMN = "co_occurrence"
NORMALIZED_COLUMN = "normalized_value"

def _extract_services(blocks_gdf: pd.DataFrame) -> (list, dict, int):
    """
    Функция для извлечения типов сервисов из столбцов, начинающихся с capacity_

    Params:
        blocks_gdf (pd.DataFrame): Датафрейм с данными о кварталах и сервисах

    Returns:
        Список всех сервисов, словарь индексов сервисов и их количество
    """
    service_columns = [col for col in blocks_gdf.columns if col.startswith('capacity_')]
    all_services = [col.replace('capacity_', '') for col in service_columns]
    service_to_idx = {s: i for i, s in enumerate(all_services)}
    n_services = len(all_services)

    return all_services, service_to_idx, n_services



def _calculate_co_occurrence(blocks_gdf: pd.DataFrame, 
                             all_services: list, 
                             service_to_idx: dict) -> np.ndarray:
    """
    Функция для вычисления матрицы совместной встречаемости сервисов

    Params:
        blocks_gdf (pd.DataFrame): Датафрейм с данными о кварталах и сервисах
        all_services (list): Список всех сервисов
        service_to_idx (dict): Словарь индексов сервисов

    Returns:
        Матрица совместной встречаемости
    """
    n_services = len(all_services)
    co_occurrence = np.zeros((n_services, n_services), dtype=int)

    for _, row in blocks_gdf.iterrows():
        services_in_block = [
            service for service in all_services
            if row[f'capacity_{service}'] > 0
        ]
        for s1, s2 in combinations(services_in_block, 2):
            i, j = service_to_idx[s1], service_to_idx[s2]
            co_occurrence[i, j] += 1
            co_occurrence[j, i] += 1

    return co_occurrence


def _calculate_block_and_total_counts(blocks_gdf: pd.DataFrame, 
                                      all_services: list) -> (dict, dict):
    """
    Функция для подсчета количества кварталов и общей емкости для каждого сервиса

    Params:
        blocks_gdf (pd.DataFrame): Датафрейм с данными о кварталах и сервисах
        all_services (list): Список всех сервисов

    Returns:
        Словарь количества кварталов и словарь общей емкости
    """
    block_counts = {
        service: (blocks_gdf[f'capacity_{service}'] > 0).sum()
        for service in all_services
    }
    total_counts = {
        service: blocks_gdf[f'capacity_{service}'].sum()
        for service in all_services
    }

    return block_counts, total_counts


def _calculate_extended_occurrence(co_occurrence: np.ndarray, 
                                   all_services: list, 
                                   service_to_idx: dict, 
                                   total_counts: dict) -> np.ndarray:
    """
    Фукнция для расширения матрицы совместной встречаемости с учетом самоколлокации

    Params:
        co_occurrence (np.ndarray): Матрица совместной встречаемости
        all_services (list): Список всех сервисов
        service_to_idx (dict): Словарь индексов сервисов
        total_counts (dict): Словарь общей емкости сервисов

    Returns:
        Расширенная матрица совместной встречаемости
    """
    extended_occurrence = co_occurrence.copy()

    for s in all_services:
        i = service_to_idx[s]
        extended_occurrence[i, i] = total_counts[s] * (total_counts[s] - 1) // 2

    return extended_occurrence


def _normalize_matrix(extended_occurrence: np.ndarray, 
                      all_services: list, 
                      service_to_idx: dict, 
                      block_counts: dict, 
                      co_occurrence: np.ndarray) -> pd.DataFrame:
    """
    Фукнция для нормализации матрицы коллокации

    Params:
        extended_occurrence (np.ndarray): Расширенная матрица совместной встречаемости
        all_services (list): Список всех сервисов
        service_to_idx (dict): Словарь индексов сервисов
        block_counts (dict): Словарь количества кварталов для каждого сервиса
        co_occurrence (np.ndarray): Матрица совместной встречаемости

    Returns:
        Нормализованная матрица коллокации
    """
    normalized_matrix = pd.DataFrame(
        np.zeros_like(extended_occurrence, dtype=float),
        index=all_services,
        columns=all_services
    )

    for s1 in all_services:
        for s2 in all_services:
            i, j = service_to_idx[s1], service_to_idx[s2]

            if s1 == s2:
                if block_counts[s1] > 1:
                    denominator = (block_counts[s1] * (block_counts[s1] - 1)) / 2
                    normalized_matrix.loc[s1, s2] = extended_occurrence[i, j] / denominator
                else:
                    normalized_matrix.loc[s1, s2] = 0  
            else:
                total_blocks = block_counts[s1] + block_counts[s2] - co_occurrence[i, j]
                if total_blocks > 0:
                    normalized_matrix.loc[s1, s2] = co_occurrence[i, j] / total_blocks

    return normalized_matrix


def _scale_diagonal(normalized_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Масштабирование диагональных значений матрицы

    Params:
        normalized_matrix (pd.DataFrame): Нормализованная матрица коллокации

    Returns:
        Матрица с масштабированными диагональными значениями
    """
    diagonal_values = normalized_matrix.values.diagonal().copy()

    if diagonal_values.max() > 0:
        scaled_diagonal = diagonal_values / diagonal_values.max()
        np.fill_diagonal(normalized_matrix.values, scaled_diagonal)

    return normalized_matrix


def collocation_matrix(blocks_gdf: pd.DataFrame) -> pd.DataFrame:
    """
    Основная функция для вычисления нормализованной матрицы коллокации

    Params:
        blocks_gdf (pd.DataFrame): Датафрейм с данными о кварталах и сервисах

    Returns:
        Нормализованная матрица коллокации
    """
    # df_schema(blocks_gdf, required_prefix="capacity_")

    all_services, service_to_idx, _ = _extract_services(blocks_gdf)
    co_occurrence = _calculate_co_occurrence(blocks_gdf, all_services, service_to_idx)
    block_counts, total_counts = _calculate_block_and_total_counts(blocks_gdf, all_services)
    extended_occurrence = _calculate_extended_occurrence(co_occurrence, all_services, service_to_idx, total_counts)
    normalized_matrix = _normalize_matrix(extended_occurrence, all_services, service_to_idx, block_counts, co_occurrence)
    normalized_matrix = _scale_diagonal(normalized_matrix)

    return normalized_matrix