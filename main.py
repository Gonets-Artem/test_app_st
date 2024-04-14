import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from catboost import CatBoostRegressor
from numpy.lib.stride_tricks import sliding_window_view


class BoxNormal:
    def __init__(self):
        self.log_norm = lambda x: np.log(x + 1)
        self.exp_norm = lambda x: np.exp(x) - 1

    def to_logs(self, x):
        return self.log_norm(x)

    def to_exps(self, x):
        return self.exp_norm(x)

log_scaler = BoxNormal()

def settings():
    # Настройка заголовка и размеров активной области
    st.set_page_config(page_title="Прогнозирование выручки от продаж", layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 400px;
            margin-left: -400px;
        }
        """,
        unsafe_allow_html=True,
    )
    st.title("Веб-сервис прогнозирования продаж на основе ML-модели")


def data_proccesing_start(df):
    #### Находим столбцы конкурентов которые есть ####
    competitors_cols = []
    for i in df.columns:
        if 'competitor' in i:
            competitors_cols.append(i)

    #### Складываем оставшиеся столбцы конкурентов ####
    df['competitors'] = 0
    for i in competitors_cols:
        df['competitors'] += df[i]

    #### Удаляем оставшиеся столбцы конкурентов ####
    df = df.drop(columns = competitors_cols)
    return df


def data_proccesing_end(df, backward_window):
    #### Считаем окна данных ####
    wind_df = df[['sickness', 'generics', 'wordstat', 'value', 'competitors']]
    window = sliding_window_view(wind_df, backward_window, 0)

    mean_sickness = []
    mean_generics = []
    mean_wordstat = []
    mean_value = []
    mean_competitors = []

    std_sickness = []
    std_generics = []
    std_wordstat = []
    std_value = []
    std_competitors = []

    for i in window:
        mean_sickness.append(np.mean(i[0]))
        mean_generics.append(np.mean(i[1]))
        mean_wordstat.append(np.mean(i[2]))
        mean_value.append(np.mean(i[3]))
        mean_competitors.append(np.mean(i[4]))

        std_sickness.append(np.std(i[0]))
        std_generics.append(np.std(i[1]))
        std_wordstat.append(np.std(i[2]))
        std_value.append(np.std(i[3]))
        std_competitors.append(np.std(i[4]))

    #### Делаем лаги для пронозируемых значений ####
    if 'count' in df:
        for i in range(1, backward_window):
            df[f'sales_shift_{i}'] = df['sales'].shift(i)
            df[f'count_shift_{i}'] = df['count'].shift(i)
    else:
        for i in range(1, backward_window):
            df[f'sales_shift_{i}'] = df['sales'].shift(i)

    df = df.dropna().reset_index(drop=True)

    #### Заполняем ДФ посчитанными значениями из окон ####
    df['mean_sickness'] = mean_sickness
    df['mean_generics'] = mean_generics
    df['mean_wordstat'] = mean_wordstat
    df['mean_value'] = mean_value
    df['mean_competitors'] = mean_competitors

    df['std_sickness'] = std_sickness
    df['std_generics'] = std_generics
    df['std_wordstat'] = std_wordstat
    df['std_value'] = std_value
    df['std_competitors'] = std_competitors

    #### Убираем лишние колонки ####
    df = df.drop(columns=['start_date', 'sickness', 'generics', 'wordstat', 'value', 'competitors'])

    return df

def post_processing(df, forward_window, xvost):

    #### Загружаем хвост известных наперед данных ####

    #### Дополняем известные данные для sample_submission ####
    if 'tv' in df:
        for i in range(1, forward_window+1):
            df[f'tv_{i}'] = df['tv'].shift(-i)
            df.loc[len(df)-1:, f'tv_{i}'] = xvost.iloc[i-1,0]
    if 'digital' in df:
        for i in range(1, forward_window+1):
            df[f'digital_{i}'] = df['digital'].shift(-i)
            df.loc[len(df)-1:, f'digital_{i}'] = xvost.iloc[i-1,1]
    if 'radio' in df:
        for i in range(1, forward_window+1):
            df[f'radio_{i}'] = df['radio'].shift(-i)
            df.loc[len(df)-1:, f'radio_{i}'] = xvost.iloc[i-1,2]

    #### Делаем y-ки для обучения ####
    for i in range(1, forward_window+1):
        df[f'y_{i}'] = df['sales'].shift(-i)

    #### Логарифмируем данные ####
    df_log = df.copy()
    df_log = df_log.dropna()
    df_log.iloc[:,2:] = log_scaler.to_logs(df_log.iloc[:,2:])

    #### Добавялем последнюю строчку для построения прогноза ####
    last = df.iloc[-1:,:-forward_window]
    last = last.fillna(0)
    last.iloc[:,2:] = log_scaler.to_logs(last.iloc[:,2:])
    
    
    return df_log, last

def forecasting(df_log, last, forward_window):
    #### Скачиваем пример submission
    sample_submission = pd.read_csv('sample_submission.csv')

    #### Делим ДФ на Х и У ####
    X = df_log.iloc[:,:-forward_window]
    y = df_log.iloc[:,-forward_window:]

    #### Определяем cat_features ####
    cats = ['week', 'year']

    #### Обучили модель ####
    model = CatBoostRegressor(loss_function='MultiRMSE', random_state = 123, cat_features=cats, l2_leaf_reg=0.5)
    model.fit(X, y, verbose=100)

    #### Сделали sample_submission ####
    sample_sub = model.predict(last)
    sample_submission['revenue'] = log_scaler.to_exps(sample_sub[0])/1.1

    #### Получили feature_importances ####
    importances = model.get_feature_importance(type='PredictionValuesChange')
    feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    #feature_importances = pd.DataFrame(feature_importances)
    
    return sample_submission, feature_importances


def main():
    settings()

    # Загрузка подготовленного файла и вывод датасета
    st.header("Источник данных")

    uploaded_file_1 = st.file_uploader("Выберите файл формата .xlsx для загрузки в веб-сервис", type="xlsx", key='123')
    if uploaded_file_1 is not None:

        df_input = pd.read_excel(uploaded_file_1, index_col=0)
        df = data_proccesing_start(df_input)
        df.drop(columns=['radio'], inplace=True)
        st.header("Выбор факторов для прогнозирования продаж")
        factors = st.multiselect('', df.columns, list(df.columns))
        st.write(df[factors])

        forward_window = st.number_input('Выберите окно в количестве дней вперед', min_value=2, step=1, value=29)
        backward_window = st.number_input('Выберите окно в количестве дней назад', min_value=4, step=1, value=58)
        if isinstance(forward_window, int) and isinstance(backward_window, int):
            df = data_proccesing_end(df, backward_window)
            
            st.header("Данные по затратам на время прогноза")
            uploaded_file_2 = st.file_uploader("Выберите файл формата .xlsx для загрузки в веб-сервис", type="xlsx", key='234')
            if uploaded_file_2 is not None:
                xvost = pd.read_excel(uploaded_file_2)
                st.write(xvost)
                df_log, last = post_processing(df, forward_window, xvost)

                st.header("Создание модели")

                if st.button("Запуск процесса прогнозирования", type="primary"):
                    st.text("Процесс обучения может занять некоторое время")

                    sample_submission, feature_importances = forecasting(df_log, last, forward_window)

                    col1, col2 = st.columns([3, 1])
                    data = pd.DataFrame(data=sample_submission.iloc[:,1])

                    col1.subheader("Прогноз")
                    col1.line_chart(round(data / 1_000_000,2), color='#DDDDDD')
                    col2.write(sample_submission)

                    col3, col4 = st.columns([3, 1])

                    col3.subheader("Наиболее важные факторы")
                    col3.bar_chart(feature_importances.nlargest(10), color='#ffffff')
                    col4.write(feature_importances)


if __name__ == "__main__":
    main()