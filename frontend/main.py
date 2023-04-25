"""
Frontend-часть проекта
Версия: 1.0
"""
import yaml
import os

import streamlit as st

from src.data.get_data import get_data, load_data
from src.plotting.charts import barplot_group, displot_category, boxplots_parts_day
from src.training.training import start_training
from src.evaluate.evaluate import evaluate_input, evaluate_from_file

st.set_option('deprecation.showPyplotGlobalUse', False)

CONFIG_PATH = "../config/parameters.yaml"


def main_page():
    """Страница с описанием проекта"""
    st.markdown("# Описание проекта")
    st.markdown("## MLOps project: User Gender Prediction cross HTTP-cookies")
    st.write("Определение пола владельца HTTP cookie по истории активности пользователя в интернете на основе "
             "синтетических данных")
    st.write("Данные взяты из соревнования от МТС Digital Big Data")
    st.write("https://ods.ai/competitions/mtsmlcup")
    st.markdown("""
                ### Описание признаков
                - **user_id** - id пользователя
                - **part_of_day_{morning/day/evening/night}** - кол-во визитов в разное время суток
                - **sum_visits** - общее кол-во визитов пользователя
                - **{morning/day/evening/night}_pct** - доля визитов в разное время суток
                - **act_days** - кол-во дней с активностью пользователя
                - **request_cnt** - кол-во запросов пользователя
                - **avg_req_per_day** - среднее кол-во запросов пользователя
                - **period_days** - кол-во дней между первым и последним визитом пользователя
                - **act_days_pct** - доля дней, когда пользователь совершал визит
                - **cpe_type_cd** - тип устройства
                - **cpe_model_os_type** - операционная система устройства
                - **cpe_manufacturer_name** - производитель устройства
                - **price** - цена устройства пользователя
                - **region_cnt** - кол-во уникальных регионов, из которых был совершен визит
                - **city_cnt** - кол-во уникальных городов, из которых был совершен визит
                
                Целевая переменная:
                - **is_male** - 1: мужчина, 0: женщина
                """)


def eda_page():
    """Exploratory data analysis"""
    st.markdown("# Exploratory data analysis")
    st.write("Исследовательский анализ аггрегированных данных")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load and write dataset
    data = get_data(data_path=config["train"]["train_data_path"])
    st.write("Аггрегированные данные:")
    st.write(data.head())

    # plotting with checkbox
    gender_os_type = st.sidebar.checkbox("Пол - Операционная система")
    gender_device_type = st.sidebar.checkbox("Пол - Тип устройства")
    gender_device_price = st.sidebar.checkbox("Пол - Цена девайса")
    gender_avg_requests = st.sidebar.checkbox("Пол - Среднее количество запросов в сутки")
    gender_part_day_pct = st.sidebar.checkbox("Пол - Доля заходов в разные части суток")

    if gender_os_type:
        st.pyplot(
            barplot_group(
                df=data,
                col_main='is_male',
                col_group='cpe_model_os_type',
                title='Пол - Операционная система'
            )
        )
        st.write("Вывод: среди пользователей Android небольшой перевес в соотношении в пользу мужчин, "
                 "среди пользователей iOS - в пользу женщин")

    if gender_device_type:
        st.pyplot(
            barplot_group(
                df=data,
                col_main='is_male',
                col_group='cpe_type_cd',
                title='Пол - Тип устройства'
            )
        )
        st.write("Вывод: 79% пользователей phablet - мужчины, в остальных категориях соотношение примерно равное "
                 "с небольшим перевесом в пользу мужчин")

    if gender_device_price:
        st.pyplot(
            displot_category(
                data=data,
                cat_feature='is_male',
                distribution_feature='price',
                plot_title='Пол - Цена девайса',
                limit=125000
            )
        )
        st.write("Вывод: в диапазоне цены девайса 50-70 тысяч небольшой перевес в пользу женщин, "
                 "в остальном распределения очень похожи")

    if gender_avg_requests:
        st.pyplot(
            displot_category(
                data=data,
                cat_feature='is_male',
                distribution_feature='avg_req_per_day',
                plot_title='Пол - Среднее количество запросов в сутки',
                limit=150
            )
        )
        st.write("Вывод: среди людей, которые отправляют запросы до 35 раз в день больше женщин, среди тех, "
                 "кто отправил запрос от 35 до 80 раз - больше мужчин, дальше примерно равное соотношение")

    if gender_part_day_pct:
        st.pyplot(
            boxplots_parts_day(
                data=data,
                cat_feature='is_male'
            )
        )
        st.write("Вывод: медианные значения доли заходов утром и ночью выше у мужчин, днем и вечером - у женщин")


def training():
    "Обучение модели"
    st.markdown("# Training model CatBoost")
    st.write("Подбод параметров, обучение модели и вывод метрик")
    # get params
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # endpoint
    endpoint = config['endpoints']['train']

    # обучение модели
    if st.button("Start training"):
        start_training(config=config, endpoint=endpoint)


def prediction_input():
    """
    Предсказание модели по введенным данным
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config['endpoints']['predict_input']
    unique_values_path = config['preprocessing']['unique_values_path']

    # проверка на наличие обученной модели
    if os.path.exists(config['train']['model_path']):
        evaluate_input(unique_values_path=unique_values_path, endpoint=endpoint)
    else:
        st.error("Необходимо обучить модель")


def prediction_from_file():
    """Получение предсказаний из файла с данными"""
    st.markdown("# Prediction")
    st.write("Получение предсказания из файла. Файл может содержать как сырые, так и аггрегированные данные")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["predict_from_file"]

    upload_file = st.file_uploader(
        "", type=["csv"], accept_multiple_files=False
    )
    # проверка загружен ли файл
    if upload_file:
        dataset_csv_df, files = load_data(data=upload_file, type_data="Test")
        # проверка на наличие сохраненной модели
        if os.path.exists(config["train"]["model_path"]):
            evaluate_from_file(data=dataset_csv_df, endpoint=endpoint, files=files)
        else:
            st.error("Необходимо обучить модель")


def main():
    """Сборка пайплайна"""
    page_names = {
        'Описание проекта': main_page,
        'Exploratory data analysis': eda_page,
        'Обучение модели': training,
        'Prediction': prediction_input,
        'Prediction from file': prediction_from_file,
    }
    selected_page = st.sidebar.selectbox("Выберите раздел", page_names.keys())
    page_names[selected_page]()


if __name__ == "__main__":
    main()
