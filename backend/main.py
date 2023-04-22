"""
Модель для предсказания пола пользователя по информации из его cookie-файлов
Версия: 1.0
"""
import warnings

import optuna
import uvicorn
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from src.preprocessing.preprocessing_input_data import preprocessing_input
from src.train.metrics import load_metrics
from src.pipeline.pipeline import pipeline_train
from src.evaluate.evaluate import evaluate_pipeline

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI()
CONFIG_PATH = "../config/parameters.yaml"


class UserCookies(BaseModel):
    """
    Признаки для получения результатов модели
    """
    part_of_day_day: int
    part_of_day_evening: int
    part_of_day_morning: int
    part_of_day_night: int
    act_days: int
    request_cnt: int
    period_days: int
    cpe_type_cd: str
    cpe_manufacturer_name: str
    price: float
    region_cnt: int
    city_cnt: int
    url_host_cnt: int


@app.post("/train")
def train():
    """
    Обучение модели и логирование метрик
    """
    pipeline_train(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH)
    return {'metrics': metrics}


@app.post("/predict")
def predict_from_file(file: UploadFile = File(...)):
    """
    Предсказание модели из файла
    """
    predictions = evaluate_pipeline(config_path=CONFIG_PATH, data_path=file.file)
    assert isinstance(predictions, list), "Результат не соответсвует типу list"
    return {"predictions": predictions[:5]}


@app.post("/predict_input")
def predict_input(user: UserCookies):
    """
    Предсказание модели по введенным данным
    """
    features = [
        [
            user.part_of_day_day,
            user.part_of_day_evening,
            user.part_of_day_morning,
            user.part_of_day_night,
            user.act_days,
            user.request_cnt,
            user.period_days,
            user.cpe_type_cd,
            user.cpe_manufacturer_name,
            user.price,
            user.region_cnt,
            user.city_cnt,
            user.url_host_cnt,
        ]
    ]

    cols = [
        'part_of_day_day',
        'part_of_day_evening',
        'part_of_day_morning',
        'part_of_day_night',
        'act_days',
        'request_cnt',
        'period_days',
        'cpe_type_cd',
        'cpe_manufacturer_name',
        'price',
        'region_cnt',
        'city_cnt',
        'url_host_cnt'
    ]

    data = pd.DataFrame(features, columns=cols)
    data = preprocessing_input(data)
    predictions = evaluate_pipeline(config_path=CONFIG_PATH, data=data)[0]
    result = (
        {"The user is male"}
        if predictions == 1
        else {"The user is female"}
        if predictions == 0
        else "Error result"
    )
    return result


if __name__ == "__main__":
    # Запустите сервер, используя заданный хост и порт
    uvicorn.run(app, host="127.0.0.1", port=80)
