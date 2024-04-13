import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def main(old):
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

    st.header("Источник данных")
    uploaded_file = st.file_uploader("Выберите файл формата .xlsx для загрузки в веб-сервис", type="xlsx")
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        
        st.header("Выбор факторов для прогнозирования продаж")
        factors = st.multiselect('', df.columns, list(df.columns))
        st.write(df[factors])

        # text_new = st.text("Изменились факторы, нужно пересоздать модель")
        # new = [el for el in factors]
        # 

        # if st.button("Подтвердить факторы", type="secondary"):
        #     if old != new:
        #         st.text(old)
        #         old = [el for el in new]
        #         st.text("Изменились фак111111111111торы, нужно пересоздать модель")
                
        #     else:
        #         st.text['111111111111111111']

        st.header("Создание модели")

        if st.button("Запуск процесса прогнозирования", type="primary"):
            st.text("Процесс обучения может занять некоторое время")

            col1, col2 = st.columns([2, 2])
            data = np.random.randn(10, 1)

            col1.subheader("A wide column with a chart")
            col1.line_chart(data)

            col2.subheader("A narrow column with the data")
            col2.write(data)


if __name__ == "__main__":
    main(old=[])