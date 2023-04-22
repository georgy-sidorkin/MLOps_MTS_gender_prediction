"""
Получение данных из файла
Версия: 1.0
"""
import pandas as pd


def get_data(data_path: str) -> pd.DataFrame:
    """
    Чтение данных по заданному пути
    :param data_path: путь до csv файла
    :return: датасет
    """
    return pd.read_csv(data_path)
