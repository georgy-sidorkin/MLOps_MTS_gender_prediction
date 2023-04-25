"""
Отравка запроса на обучение модели на backend и отображение метрик
Версия: 1.0
"""
import os
import json

import joblib
import streamlit as st
import requests
from optuna.visualization import plot_param_importances, plot_optimization_history


def start_training(config: dict, endpoint: object) -> None:
    """
    Обучение модели и вывод результатов
    :param config: словарь с данными из кофига
    :param endpoint: train endpoint
    """
    # загрузка последних метрик
    if os.path.exists(config['train']['metrics_path']):
        with open(config['train']['metrics_path']) as file:
            last_metrics = json.load(file)
    else:
        # если сохраненных метрик нет
        last_metrics = {"roc_auc": 0, "precision": 0, "recall": 0, "f1": 0}

    # обучение модели
    with st.spinner("Обучение модели..."):
        result = requests.post(endpoint, timeout=8000)
    st.success("Success ✅")

    output = result.json()
    new_metrics = output["metrics"]

    # diff metrics
    roc_auc, precision, recall, f1_metric = st.columns(4)
    roc_auc.metric(
        "ROC-AUC",
        new_metrics["roc_auc"],
        f"{new_metrics['roc_auc']-last_metrics['roc_auc']:.3f}",
    )
    precision.metric(
        "Precision",
        new_metrics["precision"],
        f"{new_metrics['precision']-last_metrics['precision']:.3f}",
    )
    recall.metric(
        "Recall",
        new_metrics["recall"],
        f"{new_metrics['recall']-last_metrics['recall']:.3f}",
    )
    f1_metric.metric(
        "F1 score", new_metrics["f1"], f"{new_metrics['f1']-last_metrics['f1']:.3f}"
    )

    # графики из optuna
    study = joblib.load(config['train']['study_path'])
    fig_imp = plot_param_importances(study)
    fig_history = plot_optimization_history(study)

    st.plotly_chart(fig_imp, use_container_width=True)
    st.plotly_chart(fig_history, use_container_width=True)
