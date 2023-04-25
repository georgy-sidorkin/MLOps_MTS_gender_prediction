"""
Функции для создания графиков
Версия: 1.0
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def barplot_group(df: pd.DataFrame, col_main: str, col_group: str,
                  title: str) -> None:
    """
    Построение barplot с нормированными данными с выводом значений на графике
    """

    plt.figure(figsize=(18, 6))

    data = (df.groupby(
        [col_group])[col_main].value_counts(normalize=True).rename(
        'percentage').mul(100).reset_index().sort_values(col_group))

    ax = sns.barplot(x=col_main,
                     y="percentage",
                     hue=col_group,
                     data=data,
                     palette='rocket')

    for p in ax.patches:
        percentage = '{:.1f}%'.format(p.get_height())
        ax.annotate(
            percentage,  # текст
            (p.get_x() + p.get_width() / 2., p.get_height()),  # координата xy
            ha='center',  # центрирование
            va='center',
            xytext=(0, 7),
            textcoords='offset points',  # точка смещения относительно координаты
            fontsize=12)

    plt.title(title, fontsize=16)
    plt.ylabel('Percentage', fontsize=14)
    plt.xlabel(col_main, fontsize=14)
    plt.show()


def displot_category(data: pd.DataFrame,
                     cat_feature: str,
                     distribution_feature: str,
                     plot_title: str,
                     limit: int = 0) -> None:
    """
    Построение displot с разбивкой по группам
    :param data: датасет
    :param cat_feature: признак для разбивки
    :param distribution_feature: признак, по которому будем смореть распределение
    :param plot_title: название графика
    :param limit: ограничение для признака распределения
    """

    values_dict = {}
    for i in data[cat_feature].unique():
        values_dict[str(i)] = i

    if limit:
        data = data[data[distribution_feature] < limit]

    sns.displot(
        {
            k: data[data[cat_feature] == v][distribution_feature]
            for k, v in values_dict.items()
        },
        kind="kde",
        common_norm=False,
        height=6,
        aspect=2.1)

    plt.title(plot_title, fontsize=16)

    plt.xlabel(distribution_feature, fontsize=14)
    plt.ylabel('Density', fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.show()


def boxplots_parts_day(
        data: pd.DataFrame,
        cat_feature: str) -> None:
    """
    Построение boxplot с разбивкой по группам и разным частям суток
    :param data: датасет
    :param cat_feature: признак для разбивки
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 14))

    graph1 = sns.boxplot(data=data,
                         x='morning_pct',
                         y=cat_feature,
                         orient='h',
                         ax=axes[0, 0])
    axes[0, 0].set_title('Доля сеансов утром', fontsize=16)
    graph1.set(xlabel=None)

    graph2 = sns.boxplot(data=data, x='day_pct', y=cat_feature, orient='h', ax=axes[0, 1])
    axes[0, 1].set_title('Доля сеансов днем', fontsize=16)
    graph2.set(xlabel=None)

    graph3 = sns.boxplot(data=data,
                         x='evening_pct',
                         y=cat_feature,
                         orient='h',
                         ax=axes[1, 0])
    axes[1, 0].set_title('Доля сеансов вечером', fontsize=16)
    graph3.set(xlabel=None)

    graph4 = sns.boxplot(data=data,
                         x='night_pct',
                         y=cat_feature,
                         orient='h',
                         ax=axes[1, 1])
    axes[1, 1].set_title('Доля сеансов ночью', fontsize=16)
    graph4.set(xlabel=None)

    plt.show()
