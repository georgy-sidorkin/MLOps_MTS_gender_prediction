"""
Получение метрик
Версия: 1.0
"""
import json
import yaml

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def get_metrics_dict(y_test: pd.Series, y_pred: np.array, y_score: np.array) -> dict:
    """
    Функция считает метрики классификации и возвращает словарь
    :param y_test: истинные значения таргета
    :param y_pred: предсказанные значения
    :param y_score: предсказанные вероятности
    :return: словарь с метриками
    """
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_score[:, 1]), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4)
    }
    return metrics


def save_metrics(
    x_data: pd.DataFrame, y_data: pd.Series, model: object, metrics_path: str
) -> None:
    """
    Получение и сохранение метрик
    :param x_data: объект-признаки
    :param y_data: целевая переменная
    :param model: модель
    :param metrics_path: путь для сохранения метрик
    """
    metrics = get_metrics_dict(
        y_test=y_data,
        y_pred=model.predict(x_data),
        y_score=model.predict_proba(x_data),
    )
    with open(metrics_path, "w") as file:
        json.dump(metrics, file)


def load_metrics(config_path: str) -> dict:
    """
    Получение метрик из файла
    :param config_path: путь до конфигурационного файла
    :return: метрики
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config["train"]["metrics_path"]) as json_file:
        metrics = json.load(json_file)

    return metrics
