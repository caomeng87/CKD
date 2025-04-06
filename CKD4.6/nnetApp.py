#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/24 16:38
# @Author  : Ken
# @Software: PyCharm
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def get_one_hot(train_data):
    # 列出分类变量的列名
    categorical_columns = ['Education level']
    # 注意：在这个示例中，'F', 'G', 'H', 'I' 在原始数据中不存在，实际使用时需要确保这些列存在

    # 提取分类变量列
    categorical_df = train_data[categorical_columns]
    categorical_df = categorical_df.astype(str)
    # 对分类变量列进行one-hot编码
    one_hot_encoded_df = pd.get_dummies(categorical_df, drop_first=False)

    # 将编码后的DataFrame与原DataFrame合并（除了分类变量列）
    result_df = pd.concat([train_data.drop(columns=categorical_columns), one_hot_encoded_df], axis=1)

    # 查看结果
    return result_df


def get_feature_date(in_df):
    raw_data = pd.read_csv('CKD.csv')

    all_data_copy = raw_data.copy()
    test_data = pd.concat([in_df, all_data_copy], axis=0)

    all_data = get_one_hot(raw_data)
    test_data = get_one_hot(test_data)

    X = all_data.drop(columns='Outcome')
    y = all_data['Outcome'].values

    x_out = test_data.drop(columns='Outcome')

    # 修改 Bool转为float类型
    X = X.astype(float)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    from imblearn.over_sampling import SMOTE

    # 检查类别分布
    print(f"Original dataset shape\n%s" % (pd.value_counts(y_train)))

    # 使用SMOTE进行过采样
    smote = SMOTE(random_state=42)
    x_train, y_train = smote.fit_resample(x_train, y_train)
    print(f"\nResampled dataset shape\n%s" % (pd.value_counts(y_train)))

    # #数值型特征 数据标准化
    standarscaler = StandardScaler()
    standarscaler.fit(x_train)

    x_out = standarscaler.transform(x_out)
    return x_out


# 标题,居中
st.markdown("<h2 style='text-align: center; color: green;'>Risk Prediction Of Cognitive Impairment In Patients With Chronic Kidney Disease </h2", unsafe_allow_html=True)

A = st.number_input('Age', max_value=200, min_value=0)
B = st.number_input('Hemoglobin concentration', max_value=30, min_value=0)
C = st.number_input('Education level', max_value=9, min_value=1)
D = st.number_input('Social participation', max_value=33, min_value=0)

Outcome = 0
input_df = pd.DataFrame([Outcome, A, B, C, D]).T
input_df.columns = ['Outcome', 'Age', 'Hemoglobin concentration', 'Education level', 'Social participation']
x_df = get_feature_date(input_df)
if st.button('Start to predict'):
    model = joblib.load('best_model.pkl')

    predictions = model.predict(x_df)
    probas = model.predict_proba(x_df)[:, 1]
    print(probas[0])
    result = round(100 * probas[0], 2)
    res = 'The risk of cognitive impairment is: ' + str(result) + '%'
    # st.subheader('The risk of cognitive impairment is: ' + str(result) + '%')
    st.markdown(f"<h4 style='text-align: center; color: red;'>{res}</h4", unsafe_allow_html=True)
