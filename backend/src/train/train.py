"""
Поиск параметров и обучение модели
Версия: 1.0
"""
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np

from ..data.train_test_split import get_split_data
from ..train.metrics import save_metrics


def objective(
        trial,
        data_x: pd.DataFrame,
        data_y: pd.Series,
        n_folds: int = 5,
        random_state: int = 10):
    """
    Функция для подбора параметров
    :param trial: кол-во trials
    :param data_x: данные с объект-признаками
    :param data_y: данные с таргетом
    :param n_folds: кол-во фолдов
    :param random_state: random state
    :return: среднее значение метрики по фолдам
    """
    cat_params = {
        "iterations":
            trial.suggest_categorical("iterations", [1300]),
        "learning_rate":
            trial.suggest_categorical("learning_rate", [0.048130802271454436]),
        "max_depth":
            trial.suggest_int("max_depth", 4, 10),
        "colsample_bylevel":
            trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "l2_leaf_reg":
            trial.suggest_uniform("l2_leaf_reg", 1e-5, 1e2),
        "random_strength":
            trial.suggest_uniform('random_strength', 10, 50),
        "bootstrap_type":
            trial.suggest_categorical("bootstrap_type",
                                      ["Bayesian", "Bernoulli", "MVS", "No"]),
        "border_count":
            trial.suggest_categorical('border_count', [128, 254]),
        "grow_policy":
            trial.suggest_categorical('grow_policy',
                                      ["SymmetricTree", "Depthwise", "Lossguide"]),
        "od_wait":
            trial.suggest_int('od_wait', 500, 2000),
        "leaf_estimation_iterations":
            trial.suggest_int('leaf_estimation_iterations', 1, 15),
        # "use_best_model":
        #     trial.suggest_categorical("use_best_model", [True]),
        "eval_metric":
            trial.suggest_categorical("eval_metric", ['AUC']),
        "random_state":
            trial.suggest_categorical("random_state", [random_state])
    }

    if cat_params["bootstrap_type"] == "Bayesian":
        cat_params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature", 0, 100)
    elif cat_params["bootstrap_type"] == "Bernoulli":
        cat_params["subsample"] = trial.suggest_float("subsample",
                                                      0.1,
                                                      1,
                                                      log=True)

    cv_folds = StratifiedKFold(n_splits=n_folds,
                               shuffle=True,
                               random_state=random_state)

    cv_predicts = np.empty(n_folds)

    for idx, (train_idx, test_idx) in enumerate(cv_folds.split(data_x, data_y)):
        x_train, x_test = data_x.iloc[train_idx], data_x.iloc[test_idx]
        y_train, y_test = data_y.iloc[train_idx], data_y.iloc[test_idx]

        cat_features = data_x.select_dtypes('category').columns.tolist()
        model = CatBoostClassifier(**cat_params, cat_features=cat_features)
        model.fit(x_train,
                  y_train,
                  eval_set=[(x_test, y_test)],
                  early_stopping_rounds=100,
                  verbose=0)

        preds_proba = model.predict_proba(x_test)[:, 1]
        cv_predicts[idx] = roc_auc_score(y_test, preds_proba)

        return np.mean(cv_predicts)


def find_optimal_params(
        data_train: pd.DataFrame, data_test: pd.DataFrame, **kwargs
) -> optuna.Study:
    """
    Пайплайн для тренировки модели
    :param data_train: датасет train
    :param data_test: датасет test
    :return: [CarBoostClassifier tuning, Study]
    """
    x_train, x_test, y_train, y_test = get_split_data(
        data_train=data_train, data_test=data_test, target=kwargs["target"]
    )

    study = optuna.create_study(direction="maximize", study_name="CatBoost")
    function = lambda trial: objective(
        trial, x_train, y_train, kwargs["k_folds"], kwargs["random_state"]
    )
    study.optimize(function, n_trials=kwargs["n_trials"], show_progress_bar=True)
    return study


def train_model(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    study: optuna.Study,
    target: str,
    metric_path: str,
) -> CatBoostClassifier:
    """
    Обучение модели на лучших параметрах
    :param data_train: тренировочный датасет
    :param data_test: тестовый датасет
    :param study: study optuna
    :param target: название целевой переменной
    :param metric_path: путь до папки с метриками
    :return: CatBoostClassifier
    """
    # разбивка данных на train/test
    x_train, x_test, y_train, y_test = get_split_data(
        data_train=data_train, data_test=data_test, target=target
    )

    # обучение на лучших параметрах
    cat_features = x_train.select_dtypes('category').columns.tolist()
    clf = CatBoostClassifier(**study.best_params,
                             allow_writing_files=False,
                             cat_features=cat_features,
                             verbose=False)
    clf.fit(x_train, y_train, verbose=False)

    # сохранение метрик
    save_metrics(x_data=x_test, y_data=y_test, model=clf, metrics_path=metric_path)
    return clf
