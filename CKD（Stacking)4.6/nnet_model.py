#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/24 15:09
# @Author  : Ken
# @Software: PyCharm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # 混淆矩阵
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import GridSearchCV

# 二分类模型各种计算指标及画图工具
from untils import *

import warnings

warnings.filterwarnings("ignore")


def get_one_hot(train_data):
    # 列出分类变量的列名
    categorical_columns = ['C']
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


class classfication():

    def __init__(self, base_clf, x_train, y_train, x_test, y_test):
        self.base_clf = base_clf
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self):
        predictions = []  # predicted labels
        actuals = []  # actual labels

        self.base_clf.fit(self.x_train, self.y_train)
        predictions = self.base_clf.predict(self.x_test)
        actuals = self.y_test
        probas = self.base_clf.predict_proba(self.x_test)[:, 1]
        return actuals, predictions, probas

    def train_score(self):
        predictions = self.base_clf.predict(self.x_train)
        actuals = self.y_train
        probas = self.base_clf.predict_proba(self.x_train)[:, 1]
        return predictions, actuals, probas

    def test_score(self, predictions, actuals):
        print(classification_report(predictions, actuals))


def train(clf, x_train, x_test, y_train, y_test):
    # 训练
    clf = classfication(clf, x_train, y_train, x_test, y_test)
    y_pred, y_test, y_prob = clf.fit()
    return clf, y_pred, y_test, y_prob


raw_data = pd.read_csv('CKD.csv')

all_data = get_one_hot(raw_data)

X = all_data.drop(columns='Outcome')
y = all_data['Outcome'].values

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

x_train = standarscaler.transform(x_train)
x_test = standarscaler.transform(x_test)
#x_out = standarscaler.transform(x_out)

n_bootstrap = 1000

from sklearn.neural_network import MLPClassifier


def mlp_gridcv(x_train, y_train):
    mlp = MLPClassifier(max_iter=100, random_state=42)  # 增加迭代次数以确保收敛

    # 定义参数网格
    param_grid = {
        'hidden_layer_sizes': [(10,)],  # 隐藏层大小
        'activation': ['tanh', 'relu'],  # 激活函数
        #'solver': ['sgd', 'adam'],  # 优化器
        'solver': ['adam'],  # 优化器
        #'alpha': [0.0001, 0.001, 0.01],  # L2惩罚项系数
        #'learning_rate': ['constant', 'adaptive'],  # 学习率调度策略
        #'learning_rate_init': [0.01, 0.05, 0.1]  # 初始学习率
        'learning_rate_init': [0.001]  # 初始学习率
    }

    # 创建GridSearchCV对象
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='roc_auc', verbose=1)

    # 训练模型并找到最佳参数
    grid_search.fit(x_train, y_train)

    # 输出最佳参数
    print("Best parameters: ", grid_search.best_params_)

    # 输出最佳模型的评分
    print("Best score on validation data: ", grid_search.best_score_)

    # 获取最佳模型
    best_mlp = grid_search.best_estimator_
    joblib.dump(best_mlp, 'best_model.pkl')
    return best_mlp

# 模型名称
name = 'NNET'
# 网格搜索最好参数
best_xgb = mlp_gridcv(x_train, y_train)

# 最优模型 预测
preds = best_xgb.predict_proba(x_test)[:, 1]
print(preds)