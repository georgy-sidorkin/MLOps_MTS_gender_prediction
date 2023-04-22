"""
Получение предсказаний с помощью предобученной модели
Версия: 1.0
"""
import os
import yaml
import joblib

import pandas as pd
from catboost import Pool

from ..data.get_data import get_data
from ..preprocessing.preprocessing_data import pipeline_preprocessing


def evaluate_pipeline(config_path: str,
                      data: pd.DataFrame = None,
                      data_path: str = None,
                      flag_raw: bool = False):

    # чтение конфигурационного файла
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    train_config = config['train']

    if data_path:
        data = get_data(data_path=data_path)

    # проверка на наличие признака из сырых данных
    # если он есть, то данные будут предобработаны, как сырые
    if 'region_name' in data.columns:
        flag_raw = True

    # обработка данных
    data = pipeline_preprocessing(data=data,
                                  cfg=config,
                                  flag_raw=flag_raw,
                                  flag_train=False)

    category_features = data.select_dtypes('category').columns.tolist()
    pool_test = Pool(data, cat_features=category_features)
    model = joblib.load(os.path.join(train_config["model_path"]))
    prediction = model.predict(pool_test).tolist()

    return prediction
