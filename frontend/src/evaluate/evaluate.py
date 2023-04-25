"""
Отрисовка слайдеров и кнопок, получение предсказаний по введенным данным
Версия: 1.0
"""
from io import BytesIO
import json
import requests
import streamlit as st
import pandas as pd


def evaluate_input(unique_values_path: str, endpoint: object) -> None:
    """
    Ввод данных, генерация новых фич и получение результатов
    :param unique_values_path: путь до файла с уникальными значениями
    :param endpoint: endpoint
    """
    with open(unique_values_path) as file:
        unique_values = json.load(file)
    # поля для ввода данных
    part_of_day_morning = st.sidebar.number_input(
        "Кол-во сессий утром",
        min_value=min(unique_values['part_of_day_morning']),
        max_value=max(unique_values['part_of_day_morning'])
    )
    part_of_day_day = st.sidebar.number_input(
        "Кол-во сессий днем",
        min_value=min(unique_values['part_of_day_day']),
        max_value=max(unique_values['part_of_day_day'])
    )
    part_of_day_evening = st.sidebar.number_input(
        "Кол-во сессий вечером",
        min_value=min(unique_values['part_of_day_evening']),
        max_value=max(unique_values['part_of_day_evening'])
    )
    part_of_day_night = st.sidebar.number_input(
        "Кол-во сессий ночью",
        min_value=min(unique_values['part_of_day_night']),
        max_value=max(unique_values['part_of_day_night'])
    )
    act_days = st.sidebar.slider(
        "Кол-во активных дней",
        min_value=min(unique_values['act_days']),
        max_value=max(unique_values['act_days'])
    )
    request_cnt = st.sidebar.number_input(
        "Кол-во запросов",
        min_value=min(unique_values['request_cnt']),
        max_value=max(unique_values['request_cnt'])
    )
    period_days = st.sidebar.slider(
        "Дней между первым и последним визитом",
        min_value=min(unique_values['period_days']),
        max_value=max(unique_values['period_days'])
    )
    cpe_type_cd = st.sidebar.selectbox(
        "Тип устройства",
        sorted(unique_values['cpe_type_cd'])
    )
    cpe_manufacturer_name = st.sidebar.selectbox(
        "Производитель устройства",
        sorted(unique_values['cpe_manufacturer_name'])
    )
    price = st.sidebar.slider(
        "Цена устройства",
        min_value=min(unique_values['price']),
        max_value=max(unique_values['price'])
    )
    region_cnt = st.sidebar.slider(
        "Кол-во регионов",
        min_value=min(unique_values['region_cnt']),
        max_value=max(unique_values['region_cnt'])
    )
    city_cnt = st.sidebar.slider(
        "Кол-во городов",
        min_value=min(unique_values['city_cnt']),
        max_value=max(unique_values['city_cnt'])
    )
    url_host_cnt = st.sidebar.slider(
        "Кол-во url",
        min_value=min(unique_values['url_host_cnt']),
        max_value=max(unique_values['url_host_cnt'])
    )

    input_dict = {
        "part_of_day_morning": part_of_day_morning,
        "part_of_day_day": part_of_day_day,
        "part_of_day_evening": part_of_day_evening,
        "part_of_day_night": part_of_day_night,
        "act_days": act_days,
        "request_cnt": request_cnt,
        "period_days": period_days,
        "cpe_type_cd": cpe_type_cd,
        "cpe_manufacturer_name": cpe_manufacturer_name,
        "price": price,
        "region_cnt": region_cnt,
        "city_cnt": city_cnt,
        "url_host_cnt": url_host_cnt
    }

    st.write(
        f""" Получение предсказания с помощью ручного ввода данных\n
        
    **Данные пользователя:**\n
    1) Кол-во сессий утром: {input_dict['part_of_day_morning']}
    2) Кол-во сессий днем: {input_dict['part_of_day_day']}
    3) Кол-во сессий вечером: {input_dict['part_of_day_evening']}
    4) Кол-во сессий ночью: {input_dict['part_of_day_night']}
    5) Кол-во активных дней: {input_dict['act_days']}
    6) Кол-во запросов: {input_dict['request_cnt']}
    7) Дней между первым и последним визитом: {input_dict['period_days']}
    8) Тип устройства: {input_dict['cpe_type_cd']}
    9) Производитель устройства: {input_dict['cpe_manufacturer_name']}
    10) Цена устройства: {input_dict['price']}
    11) Кол-во регионов: {input_dict['region_cnt']}
    12) Кол-во городов: {input_dict['city_cnt']}
    13) Кол-во url: {input_dict['url_host_cnt']}
    """
    )

    # evaluate and return prediction (text)
    button_ok = st.button("Predict")
    if button_ok:
        sum_visits = (input_dict['part_of_day_morning']
                      + input_dict['part_of_day_day']
                      + input_dict['part_of_day_evening']
                      + input_dict['part_of_day_night']
                      )
        if sum_visits == 0:
            st.error("Кол-во сессий должно быть больше 0")
        elif sum_visits > input_dict['request_cnt']:
            st.error("Кол-во визитов не может быть больше кол-ва запросов")
        elif input_dict['act_days'] > input_dict['period_days']:
            st.error("Кол-во активных дней не может быть больше кол-ва дней "
                     "между первым и последним визитом")
        else:
            result = requests.post(endpoint, timeout=8000, json=input_dict)
            json_str = json.dumps(result.json())
            output = json.loads(json_str)
            st.write(f"## {output[0]}")
            st.success("Success!")


def evaluate_from_file(data: pd.DataFrame, endpoint: object, files: BytesIO) -> None:
    """
    Получение файла в признаками и вывод предсказаний в виде таблицы
    :param data: датасет
    :param endpoint: endpoint
    :param files:
    """
    button_ok = st.button("Predict")
    if button_ok:
        dataset = data[:5]
        output = requests.post(endpoint, files=files, timeout=8000)
        df = pd.DataFrame.from_records(json.loads(output.text)['predictions'])
        st.write("Predictions:")
        st.write(df[['user_id', 'predict']].head())
