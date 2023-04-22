"""
Предобработка данных
Версия: 1.0
"""
import warnings
import math
import json

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def check_columns(data: pd.DataFrame, columns_types: dict) -> None:
    """
    Проверка соответствия колонок
    :param data: датасет
    :param columns_types: словарь с признаками и типами
    """
    column_sequence = columns_types.keys()

    assert set(column_sequence) == set(data.columns), f"Признаки не совпадают: {set(data.columns)}"


def change_cols_type(data: pd.DataFrame, col_types_dict: dict) -> pd.DataFrame:
    """
    Изменение типа столбцов на заданные
    :param data: датафрейм
    :param col_types_dict: словарь с признаками и типами данных
    :return: датафрейм
    """
    return data.astype(col_types_dict)


def fill_na_values(data: pd.DataFrame, fill_na_val: dict) -> pd.DataFrame:
    """
    Заполнение пропусков заданными значениями
    :param data: датафрейм
    :param fill_na_val: словарь с названиями признаков и значением, которым нужно заполнить пропуки
    :return: датафрейм
    """
    return data.fillna(fill_na_val)


def replace_model_mistakes(data: pd.DataFrame,
                           replace_val: dict) -> pd.DataFrame:
    """
    Функция исправляет неточности в данных
    :param data: датафрейм с данными
    :param replace_val: словарь с признаками и значениями
    :return: датафрейм
    """
    return data.replace(replace_val)


def replace_nokia_type(data: pd.DataFrame) -> pd.DataFrame:
    """
    Функция исправляет неточности в данных, заменяет
    :param data: датафрейм с данными
    :return: датафрейм с исправленными неточностями
    """
    data['cpe_type_cd'] = np.where((data['cpe_manufacturer_name'] == 'Nokia') &
                                   (data['cpe_model_name'] == '3 Dual'),
                                   'plain', data['cpe_type_cd'])
    return data


# функции для генерации признаков
def get_data_part_day(data: pd.DataFrame) -> pd.DataFrame:
    """
    Функция аггрегирует данные по user_id и возвращает долю и кол-во визитов в разное время суток
    :param data: датафрейм с данными
    :return: аггрегированный датафрейм
    """
    df_part_day = pd.get_dummies(data[['user_id', 'part_of_day']])

    # кол-во визитов пользователя в разные части дня
    df_part_day = (df_part_day.groupby('user_id', as_index=False).agg({
        'part_of_day_day': 'sum',
        'part_of_day_evening': 'sum',
        'part_of_day_morning': 'sum',
        'part_of_day_night': 'sum'
    }))

    # общее кол-во визитов пользователя
    df_part_day['sum_visits'] = (df_part_day.part_of_day_day +
                                 df_part_day.part_of_day_evening +
                                 df_part_day.part_of_day_morning +
                                 df_part_day.part_of_day_night)

    # доля визитов в разные части дня
    df_part_day['day_pct'] = df_part_day.part_of_day_day / \
                             df_part_day.sum_visits
    df_part_day['evening_pct'] = df_part_day.part_of_day_evening / \
                                 df_part_day.sum_visits
    df_part_day['morning_pct'] = df_part_day.part_of_day_morning / \
                                 df_part_day.sum_visits
    df_part_day['night_pct'] = df_part_day.part_of_day_night / \
                               df_part_day.sum_visits
    return df_part_day


def get_data_days(data: pd.DataFrame) -> pd.DataFrame:
    """
    Функция аггрегирует данные по user_id и возвращает следующие признаки:
    - act_days - кол-во дат активности пользователя
    - request_cnt - кол-во запросов пользователя
    - avg_req_per_day - среднее кол-во запросов пользователя
    - period_days - кол-во дней между первым и последним визитом пользователя
    - act_days_pct - доля дней, когда пользователь совершал визит

    :param data: датафрейм с данными
    :return: аггрегированный датафрейм
    """
    # кол-во дней с визитами
    df_active_days = (data.groupby('user_id', as_index=False).agg({
        'date':
            'nunique',
        'request_cnt':
            'sum'
    }).rename(columns={'date': 'act_days'}))

    # среднее кол-во запросов в дни визита
    df_active_days['avg_req_per_day'] = (df_active_days.request_cnt /
                                         df_active_days.act_days)

    # первая и последняя дата визита
    df_dates_period = (data.groupby('user_id', as_index=False).agg(
        {'date': ['max', 'min']}))

    # кол-во дней между первым и последним заходом
    df_dates_period['period_days'] = (df_dates_period['date'].iloc[:, 0] -
                                      df_dates_period['date'].iloc[:, 1])
    df_dates_period['period_days'] = df_dates_period.period_days.dt.days + 1
    df_dates_period = df_dates_period.drop('date', axis=1)

    df_dates_period.columns = df_dates_period.columns.droplevel(1)

    df_days = df_active_days.merge(df_dates_period, on='user_id')
    # доля дней, когда пользователь совершал визит
    df_days['act_days_pct'] = df_days.act_days / df_days.period_days

    return df_days


def get_user_model_price(data: pd.DataFrame) -> pd.DataFrame:
    """
    Функция аггрегирует данные по user_id и возвращает следующие признаки:
    - cpe_type_cd - тип устройства
    - cpe_model_os_type - операционная система устройства
    - cpe_manufacturer_name - производитель устройства
    - price - цена устройства пользователя

    :param data: датафрейм с данными
    :return: аггрегированный датафрейм
    """
    df_model = data.groupby('user_id', as_index=False).agg({
        'cpe_type_cd': pd.Series.mode,
        'cpe_manufacturer_name': pd.Series.mode,
        'cpe_model_os_type': pd.Series.mode,
        'price': 'mean'
    })
    return df_model.fillna(-999)


def get_user_city_cnt(data: pd.DataFrame) -> pd.DataFrame:
    """
    Функция аггрегирует данные по user_id и возвращает следующие признаки:
    - region_cnt - кол-во уникальных регионов, из которых был совершен визит
    - city_cnt - кол-во уникальных городов, из которых был совершен визит

    :param data: датафрейм с данными
    :return: аггрегированный датафрейм
    """
    df_city_cnt = data.groupby('user_id', as_index=False) \
        .agg({'region_name': 'nunique',
              'city_name': 'nunique'}) \
        .rename(columns={'region_name': 'region_cnt',
                         'city_name': 'city_cnt'})
    return df_city_cnt


def get_user_url_cnt(data: pd.DataFrame) -> pd.DataFrame:
    """
    Функция аггрегирует данные по user_id и возвращает следующие признаки:
    - url_host_cnt - кол-во уникальных ссылок, с которых был совершен визит

    :param data: датафрейм с данными
    :return: аггрегированный датафрейм
    """
    df_url_cnt = data.groupby('user_id', as_index=False) \
        .agg({'url_host': 'nunique'}) \
        .rename(columns={'url_host': 'url_host_cnt'})
    return df_url_cnt


def save_unique_train_data(data: pd.DataFrame,
                           columns_save_unique: list,
                           columns_save_min_max: list,
                           unique_values_path: str) -> None:
    """
    Сохранение словаря с признаками и уникальными значениями
    :param data: датасет
    :param columns_save_unique: признаки, для которых нужно сохранить уникальные значения
    :param columns_save_min_max: признаки, для которых нужно сохранить только мин и макс
    :param unique_values_path: путь до файла со словарем
    :return: None
    """
    unique_df = data[columns_save_unique]
    min_max_df = data[columns_save_min_max]

    # создаем словарь с уникальными значениями для вывода в UI
    dict_unique = {
        key: unique_df[key].unique().tolist()
        for key in unique_df.columns
    }

    # создаем словарь со значениями min, max
    dict_min_max = {
        key: [
            math.floor(min_max_df[key].min())
            if min_max_df[key].min() > 0 else 0,
            math.ceil(min_max_df[key].max())
        ]
        for key in min_max_df.columns
    }

    # объединяем словари
    dict_unique.update(dict_min_max)

    with open(unique_values_path, "w") as file:
        json.dump(dict_unique, file)


# пайплайны
def pipeline_raw_preprocessing(data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Функция обрабатывает сырые данные
    :param data: датасет с сырыми данными
    :param cfg: словарь с данными из конфигурационного файла
    :return: датасет
    """
    # проверка на соответствие признаков
    check_columns(data, cfg['preprocessing']['change_col_types'])
    # преобразование типов
    data = change_cols_type(data, cfg['preprocessing']['change_col_types'])
    # заполнение пропусков
    data = fill_na_values(data, cfg['preprocessing']['columns_fill_na'])
    # замена ошибок в данных
    data = replace_model_mistakes(data, cfg['preprocessing']['replace_values'])
    data = replace_nokia_type(data)

    return data


def pipeline_feature_generation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Функция аггрегирует сырые данные и создает необходимые признаки
    """
    # кол-во визитов пользователя в разное время суток
    data_part_day = get_data_part_day(data)
    # кол-во активных дней и среднее кол-во запросов в сутки
    data_days = get_data_days(data)
    # данные об устройстве
    data_user_model = get_user_model_price(data)
    # кол-во регионов и городов
    data_city_cnt = get_user_city_cnt(data)
    # кол-во уникальных url
    data_url_cnt = get_user_url_cnt(data)

    # объединяем все в один датафрейм
    data_final = (data_part_day
                  .merge(data_days, how='left', on='user_id')
                  .merge(data_user_model, how='left', on='user_id')
                  .merge(data_city_cnt, how='left', on='user_id')
                  .merge(data_url_cnt, how='left', on='user_id'))
    return data_final


# итоговый пайплайн
def pipeline_preprocessing(data: pd.DataFrame,
                           cfg: dict,
                           flag_raw: bool = False,
                           flag_train: bool = False) -> pd.DataFrame:
    """
    Пайплайн для предобработки данных
    :param data: датасет
    :param cfg: словарь с конфигурационными данными
    :param flag_raw: если True - данные сырые и их нужно предобработать
    :param flag_train: если True - нужно соединить признаки с таргетом
        и сохранить уникальные значения
    :return: датасет
    """
    # если данные сырые
    if flag_raw:
        # обработка сырых данных
        data = pipeline_raw_preprocessing(data, cfg)
        # генерация признаков
        data = pipeline_feature_generation(data)

    # проверка столбцов аггрегированных данных
    check_columns(data, cfg['preprocessing']['agg_columns_type'])
    # изменение типов колонок на нужные
    data = change_cols_type(data, cfg['preprocessing']['agg_columns_type'])

    # если данные для тренировки
    if flag_train:
        # сохраним уникальные значения
        save_unique_train_data(
            data=data,
            columns_save_unique=cfg['preprocessing']['columns_save_unique'],
            columns_save_min_max=cfg['preprocessing']['columns_save_min_max'],
            unique_values_path=cfg['preprocessing']['unique_values_path'])

        # загрузим таргет и удалим пропуски
        targets = pd.read_parquet(cfg['train']['target_data_path'])
        targets = targets[['user_id', cfg['train']['target']]]
        targets = targets[targets[cfg['train']['target']] != 'NA'].dropna()
        # объединим признаки и таргет
        data = data.merge(targets, on='user_id')
        # изменим тип колонки для таргета
        data = change_cols_type(data, cfg['train']['target_type'])

    return data
