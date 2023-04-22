"""
Конвейер для обучения модели
Версия: 1.0
"""
import os
import yaml
import joblib

from ..data.get_data import get_data
from ..preprocessing.preprocessing_data import pipeline_preprocessing
from ..data.train_test_split import split_data
from ..train.train import find_optimal_params, train_model


def pipeline_train(config_path: str) -> None:
    """
    Функция считывает конфиг, получает и обрабатывает данные, ищет лучшие параметры,
    обучает на них модель и сохраняет ее
    :param config_path: путь до конфигурационного файла
    :return: None
    """
    # чтение конфигурационного файла
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preproc_config = config['preprocessing']
    train_config = config['train']

    # получение данных
    data = get_data(data_path=preproc_config['agg_data_path'])

    # обработка данных
    train_data = pipeline_preprocessing(data=data, cfg=config, flag_raw=False, flag_train=True)

    # сплит данных на train/test
    df_train, df_test = split_data(train_data, **train_config)

    # поиск лучших параметров
    study = find_optimal_params(data_train=df_train, data_test=df_test, **train_config)

    # обучаем модель на лучших найденных параметрах
    cat_clf = train_model(data_train=df_train,
                          data_test=df_test,
                          study=study,
                          target=train_config['target'],
                          metric_path=train_config['metrics_path'])

    # сохраняем модель и study
    joblib.dump(cat_clf, os.path.join(train_config["model_path"]))
    joblib.dump(study, os.path.join(train_config["study_path"]))
