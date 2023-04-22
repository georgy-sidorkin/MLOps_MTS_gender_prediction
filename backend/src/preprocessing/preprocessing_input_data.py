"""
Предобработка введенных данных
Версия: 1.0
"""
import pandas as pd


def preprocessing_input(data: pd.DataFrame) -> pd.DataFrame:
    """
    Функция создает недостающие признаки из введенных пользователем
    :param data: датасет для предсказания
    :return: датасет
    """
    # кол-во визитов
    data['sum_visits'] = (data.part_of_day_day
                          + data.part_of_day_evening
                          + data.part_of_day_morning
                          + data.part_of_day_night)
    # доля визитов в разное время суток
    data['day_pct'] = data.part_of_day_day / data.sum_visits
    data['evening_pct'] = data.part_of_day_evening / data.sum_visits
    data['morning_pct'] = data.part_of_day_morning / data.sum_visits
    data['night_pct'] = data.part_of_day_evening / data.sum_visits
    # среднее кол-во запросов в сутки
    data['avg_req_per_day'] = data.request_cnt / data.act_days
    # доля дней с визитам
    data['act_days_pct'] = data.act_days / data.request_cnt
    # операционная система
    data['cpe_model_os_type'] = data.cpe_manufacturer_name.transform(lambda x: "iOS" if x == "Apple" else "Android")
    data['user_id'] = 0
    return data
