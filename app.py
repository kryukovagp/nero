# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns

# Определение функции Focal Loss
def focal_loss(gamma=2., alpha=.25, num_classes=4):
    """
    Функция Focal Loss для многоклассовой классификации.
    
    Параметры:
    - gamma: Параметр фокусировки.
    - alpha: Балансировка классов.
    - num_classes: Количество классов.
    """
    def focal_loss_fixed(y_true, y_pred):
        # One-hot кодирование
        y_true_one_hot = tf.one_hot(tf.cast(y_true, 'int32'), depth=num_classes)
        
        # Клиппинг предсказаний для предотвращения log(0)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        # Кросс-энтропия
        cross_entropy = -y_true_one_hot * K.log(y_pred)
        
        # Весовые коэффициенты
        weight = alpha * K.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        
        # Суммирование потерь по классам
        return K.sum(loss, axis=1)
    
    return focal_loss_fixed

# Загрузка предобработчика
@st.cache_resource
def load_preprocessor():
    try:
        preprocessor = joblib.load('preprocessor.pkl')
        st.success("Предобработчик успешно загружен.")
        return preprocessor
    except Exception as e:
        st.error(f"Ошибка при загрузке предобработчика: {e}")
        st.stop()

# Загрузка LabelEncoder
@st.cache_resource
def load_label_encoder():
    try:
        le_age = joblib.load('le_age.pkl')
        st.success("LabelEncoder успешно загружен.")
        return le_age
    except Exception as e:
        st.error(f"Ошибка при загрузке LabelEncoder: {e}")
        st.stop()

# Загрузка модели с кастомной функцией потерь
@st.cache_resource
def load_model_custom():
    try:
        # Загрузка LabelEncoder для определения количества классов
        le_age = load_label_encoder()
        num_classes = len(le_age.classes_)
        
        # Определение функции Focal Loss с правильным количеством классов
        loss_function = focal_loss(gamma=2., alpha=.25, num_classes=num_classes)
        
        # Загрузка модели с пользовательской функцией потерь
        model = tf.keras.models.load_model('age_group_classifier.h5',
                                          custom_objects={'focal_loss_fixed': loss_function})
        st.success("Модель успешно загружена.")
        return model
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        st.stop()

# Импортируем модуль cli из web под именем stcli
from streamlit.web import cli as stcli
# Импортируем модуль sys
import sys
# Из модуля streamlit импортируем модуль runtime
from streamlit import runtime

# Проверяем существование runtime
runtime.exists()

# Основная функция приложения
def main():
    st.title("Классификация клиентов на возрастные группы")
    st.write("Введите данные клиента для определения его возрастной группы.")

    # Загрузка предобработчика и LabelEncoder
    preprocessor = load_preprocessor()
    le_age = load_label_encoder()

    # Загрузка модели
    model = load_model_custom()

    with st.form("input_form"):
        st.subheader("Введите данные клиента")

        # Образование
        education_options = ['Basic', '2n Cycle', 'Graduation', 'Master', 'PhD']
        education = st.selectbox("Education", education_options)

        # Семейное положение
        marital_status_options = ['Single', 'Together', 'Married', 'Divorced', 
                                  'Widow', 'Alone', 'Absurd', 'YOLO']
        marital_status = st.selectbox("Marital Status", marital_status_options)

        # Годовой доход (в тысячах долларов)
        income = st.number_input("Income (k$)", min_value=0, max_value=200, value=50, step=1)

        # Количество детей дома
        kidhome = st.number_input("Kidhome", min_value=0, max_value=10, value=0, step=1)

        # Количество подростков дома
        teenhome = st.number_input("Teenhome", min_value=0, max_value=10, value=0, step=1)

        # Recency (количество дней с последней покупки)
        recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30, step=1)

        # Количество веб-посещений в месяц
        num_web_visits_month = st.number_input("NumWebVisitsMonth", min_value=0, max_value=100, value=10, step=1)

        # Время с момента регистрации (в днях)
        customer_for = st.number_input("Customer_For (days since registration)", min_value=0, max_value=5000, value=1000, step=1)

        # Суммарные расходы
        total_spend = st.number_input("TotalSpend", min_value=0, max_value=10000, value=500, step=10)

        # Общее количество покупок
        total_purchases = st.number_input("TotalPurchases", min_value=0, max_value=1000, value=50, step=1)

        # Кнопка для отправки формы
        submitted = st.form_submit_button("Classify")

    if submitted:
        # Создание DataFrame из введённых данных
        input_data = pd.DataFrame({
            'Education': [education],
            'Marital_Status': [marital_status],
            'Income': [income],
            'Kidhome': [kidhome],
            'Teenhome': [teenhome],
            'Recency': [recency],
            'NumWebVisitsMonth': [num_web_visits_month],
            'Customer_For': [customer_for],
            'TotalSpend': [total_spend],
            'TotalPurchases': [total_purchases]
        })

        # Предобработка данных
        try:
            input_preprocessed = preprocessor.transform(input_data)
        except Exception as e:
            st.error(f"Ошибка при предобработке данных: {e}")
            st.stop()

        # Предсказание
        try:
            prediction = model.predict(input_preprocessed)
            predicted_class = np.argmax(prediction, axis=1)
            age_group = le_age.inverse_transform(predicted_class)[0]
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
            st.stop()

        # Получение вероятностей
        probabilities = prediction[0]

        # Отображение результата
        st.success(f"Предсказанная возрастная группа: **{age_group}**")
        st.write("Вероятности для каждой возрастной группы:")
        proba_df = pd.DataFrame(probabilities.reshape(1, -1), columns=le_age.classes_)
        st.dataframe(proba_df * 100)

        # Визуализация вероятностей
        plt.figure(figsize=(8,6))
        sns.barplot(x=proba_df.columns, y=proba_df.iloc[0]*100, palette='viridis')
        plt.xlabel('Age Group')
        plt.ylabel('Probability (%)')
        plt.title('Probability Distribution')
        plt.ylim(0, 100)
        st.pyplot(plt)

        # Примеры ввода
        st.subheader("Примеры ввода:")
        st.write("""
        Ниже приведены примеры вводимых данных для различных возрастных групп.

        **Пример 1:**
        - **Education:** Master
        - **Marital Status:** Married
        - **Income (k$):** 80
        - **Kidhome:** 2
        - **Teenhome:** 1
        - **Recency (days since last purchase):** 20
        - **NumWebVisitsMonth:** 15
        - **Customer_For (days):** 1500
        - **TotalSpend:** 1200
        - **TotalPurchases:** 60

        **Пример 2:**
        - **Education:** Graduation
        - **Marital Status:** Divorced
        - **Income (k$):** 50
        - **Kidhome:** 3
        - **Teenhome:** 2
        - **Recency (days since last purchase):** 5
        - **NumWebVisitsMonth:** 30
        - **Customer_For (days):** 5000
        - **TotalSpend:** 1800
        - **TotalPurchases:** 100

        **Пример 3:**
        - **Education:** PhD
        - **Marital Status:** Married
        - **Income (k$):** 120
        - **Kidhome:** 1
        - **Teenhome:** 0
        - **Recency (days since last purchase):** 10
        - **NumWebVisitsMonth:** 25
        - **Customer_For (days):** 3000
        - **TotalSpend:** 2500
        - **TotalPurchases:** 80
        """)

# Если скрипт запускается напрямую
if __name__ == '__main__':
    # Если runtime существует
    if runtime.exists():
        # Вызываем функцию main()
        main()
    # Если runtime не существует
    else:
        # Устанавливаем аргументы командной строки
        sys.argv = ["streamlit", "run", sys.argv[0]]
        # Выходим из программы с помощью функции main() из модуля stcli
        sys.exit(stcli.main())
