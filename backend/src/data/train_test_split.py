"""
Разбивка данных на train/test
Версия: 1.0
"""
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разделение данные на train/test с сохранением
    :param data: датасет
    :return: train и test датасеты
    """
    data_train, data_test = train_test_split(
        data,
        stratify=data[kwargs['target']],
        test_size=kwargs['train_test_size'],
        random_state=kwargs['random_state']
    )
    data_train.to_csv(kwargs['train_split_path'], index_label=False)
    data_test.to_csv(kwargs['test_split_path'], index_label=False)
    return data_train, data_test


def get_split_data(data_train: pd.DataFrame,
                   data_test: pd.DataFrame,
                   target: str
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Разбивка данных train/test на объекты-признаки и таргет
    :param data_train: train датасет
    :param data_test: test датасет
    :param target: целевая переменная
    :return: train/test набор данных
    """
    x_train = data_train.drop(target, axis=1)
    x_test = data_test.drop(target, axis=1)

    y_train = data_train.loc[:, target]
    y_test = data_test.loc[:, target]
    return x_train, x_test, y_train, y_test
