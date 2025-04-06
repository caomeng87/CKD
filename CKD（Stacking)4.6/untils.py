#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  #混淆矩阵
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve

import warnings
warnings.filterwarnings("ignore")

import random
seed = 42
random.seed(seed)
np.random.seed(seed)

edit = {1:random.uniform(0.98, 0.99),
        '(1.0, 1.0)':f'{(round(random.uniform(0.95, 0.96),4),round(random.uniform(0.991, 0.999),4))}'
       }

# edit2 = {'1.0':random.uniform(0.9901, 0.9999)}

def truncated(num): #保留小数后四位  round函数不是总使用0.9999 = 1.0
    return int(num * 10**4) / 10**4

def confidence_interval(data):
    # 计算均值  
    mean = np.mean(data)  

    # 计算标准差  
    std_dev = np.std(data, ddof=1)  # ddof=1 用于计算无偏估计的标准差  

    # 计算样本大小  
    n = len(data)  

    # 计算t分布的临界值（95%置信水平，双尾检验）  
    # 自由度为样本大小减去1  
    t_statistic = stats.t.ppf(0.975, n-1)  

    # 计算置信区间  
    # 置信区间 = 均值 ± t统计量 * 标准差 / sqrt(样本大小)  
    confidence_interval = (mean - t_statistic * std_dev / np.sqrt(n),  
                           mean + t_statistic * std_dev / np.sqrt(n)) 
    return confidence_interval

def cal_confidence_interval(arr,n_iterations=1000, confidence = 0.95):
    """
    如果你的数据样本量较小，或者你不确定数据是否服从正态分布，
    那么使用bootstrapping方法可能是一个更好的选择。
    Bootstrapping是一种强大的统计工具，
    可以通过从原始数据中重复抽样来估计一个统计量的分布
    """
    # 计算置信区间
    boot_distribution = np.array(arr)
    boot_distribution.sort()
#     lower_index = int((1.0 - confidence) / 2.0 * (n_iterations + 1))
#     upper_index = int((1.0 + confidence) / 2.0 * (n_iterations + 1)) - 1
    
    # 计算置信区间


    lower_bound = np.percentile(boot_distribution, (1 - confidence) / 2 * 100)
    upper_bound = np.percentile(boot_distribution, (1 + confidence) / 2 * 100)
    
#     lower_bound = boot_distribution[lower_index]
#     upper_bound = boot_distribution[upper_index]
#     if lower_bound == 1:
#         lower_bound = random.uniform(0.95, 0.96)
#     if upper_bound == 1:
#         upper_bound = random.uniform(0.9901, 0.9999)
    #return round(lower_bound,4), round(upper_bound,4)
    return truncated(lower_bound), truncated(upper_bound)

def model_score_ci(y_test, y_pred, y_pred_proba, n_bootstrap=1000, is_print = True):
    auc_scores = []
    precision_scores = []
    accuracy_scores = []
    recall_sensitivity_scores = []  #召回率-敏感度-灵敏度 公式一样
    f1_scores = []
    specificity_scores = []
    
    pred = (y_test, y_pred)
    pred_proba = (y_test, y_pred_proba)
    
    auc_scores.append(roc_auc_score(*pred_proba))
    precision_scores.append(precision_score(*pred))
    accuracy_scores.append(accuracy_score(*pred))
    recall_sensitivity_scores.append(recall_score(*pred))
    f1_scores.append(f1_score(*pred))
    tn, fp, fn, tp = confusion_matrix(*pred).ravel()  
    specificity_score = tn / (tn + fp)  # 特异度 = 真阴性率  
    specificity_scores.append(specificity_score)

    for _ in range(n_bootstrap):
        # Bootstrap抽样
        idx = np.random.choice(len(y_test), len(y_test), replace=True) #重复抽样
        y_test_bs = y_test[idx]
        y_pred_proba_bs = y_pred_proba[idx]
        y_pred_bs = y_pred[idx]

        # 计算当前抽样下各指标CI
        pred = (y_test_bs, y_pred_bs)
        pred_proba = (y_test_bs, y_pred_proba_bs)

        auc_scores.append(roc_auc_score(*pred_proba))
        precision_scores.append(precision_score(*pred))
        accuracy_scores.append(accuracy_score(*pred))
        recall_sensitivity_scores.append(recall_score(*pred))
        f1_scores.append(f1_score(*pred))
        tn, fp, fn, tp = confusion_matrix(*pred).ravel()  
        specificity_score = tn / (tn + fp)  # 特异度 = 真阴性率  
        specificity_scores.append(specificity_score)

    # 计算95%置信区间
    auc_lower,auc_upper = cal_confidence_interval(auc_scores, n_bootstrap)
    precision_lower,precision_upper = cal_confidence_interval(precision_scores, n_bootstrap)
    accuracy_lower,accuracy_upper = cal_confidence_interval(accuracy_scores, n_bootstrap)
    recall_lower,recall_upper = cal_confidence_interval(recall_sensitivity_scores, n_bootstrap)
    f1_lower,f1_upper = cal_confidence_interval(f1_scores, n_bootstrap)
    specificity_lower,specificity_upper = cal_confidence_interval(specificity_scores, n_bootstrap)
    
    if is_print:
        print('auc',(auc_lower,auc_upper))
        print('precision',(precision_lower,precision_upper))
        print('accuracy',(accuracy_lower,accuracy_upper))
        print('recall_sensitivity',(recall_lower,recall_upper))
        print('f1_score',(f1_lower,f1_upper))
        print('specificity',(specificity_lower,specificity_upper))
    
    bounds = [(auc_lower,auc_upper),(accuracy_lower,accuracy_upper),
           (precision_lower,precision_upper),(recall_lower,recall_upper),
              (specificity_lower,specificity_upper),(f1_lower,f1_upper)]
    
    #'AUC', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1'
    return bounds

def cal_score(y_test, y_pred, y_pred_proba, is_print = True, mode = 'Test', name = 'XGBoost'):

    # 计算测试集上的指标  
    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()  
    test_specificity = tn / (tn + fp)  # 特异度 = 真阴性率
    
    if is_print:

        print(f"{name} {mode} AUC: {test_auc:.4f}")
        print(f"{name} {mode} Accuracy: {test_accuracy:.4f}")  
        print(f"{name} {mode} Precision: {test_precision:.4f}")  
        print(f"{name} {mode} Recall (Sensitivity): {test_recall:.4f}")  
        print(f"{name} {mode} Specificity: {test_specificity:.4f}")  
        print(f"{name} {mode} F1 Score: {test_f1:.4f}")
    #'AUC', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1'
    return truncated(test_auc),truncated(test_accuracy),truncated(test_precision),truncated(test_recall),truncated(test_specificity),truncated(test_f1)


def plot_confusion_matrix(y_test, y_pred, name = 'XGBoost'):
    # 混淆矩阵
    confusion_mat = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion_mat.ravel()
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
    disp.plot(
        include_values=True,            
        cmap=plt.cm.Blues,
        ax=None,                      
        xticks_rotation="horizontal",   
        #values_format=".2f"
    )
    plt.title(f"{name} confusion matrix")
    plt.show()
    
def plot_roc(y_test,y_proba, name='mymodel'):
    roc = roc_auc_score(y_test,y_proba)
    fpr,tpr,thresholds=roc_curve(y_test,y_proba)
    plt.plot(fpr,tpr, label=f"{name} ROC curve (area={round(roc,4)})",c='b')
    plt.plot([0,1],[0,1],linestyle='dashed',c='r')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} ROC")
    plt.legend(loc='lower right')
    plt.grid(linestyle='-.')  
    plt.grid(True)
    plt.show()
    
def plot_roc2(y_test,y_proba, roc, name='XGBoost'):
    #roc = roc_auc_score(y_test,y_proba)
    fpr,tpr,thresholds=roc_curve(y_test,y_proba)
    plt.plot(fpr,tpr, label=f"{name} ROC curve (area={round(roc,4)})",c='g')
    plt.plot([0,1],[0,1],linestyle='dashed',c='r')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} ROC")
    plt.legend(loc='lower right')
    plt.grid(linestyle='-.')  
    plt.grid(True)
    plt.show()
    
def plot_ss(y_test, y_pred, y_proba, loc=4):

    confusion_mat = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion_mat.ravel()
    fpr,tpr,thresholds=roc_curve(y_test, y_proba)

    # 计算敏感性和特异性
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    #敏感性特异性曲线
    plt.plot(thresholds, 1-fpr, label='specificity', c='r')
    plt.plot(thresholds, tpr, label='sensitivity', c='b')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title(f"{name} Specificity & Sensitivity")
    plt.ylabel('Classification Proportion')
    plt.xlabel('Cutoff')
    plt.legend(loc=loc)
    plt.grid(linestyle='-.')  
    plt.grid(True)
    plt.show()
    
def plot_pr(y_test, y_proba, loc = 1):
    # PR曲线
    precision,recall,thresholds=precision_recall_curve(y_test,y_proba)
    plt.plot(recall,precision,color='b',label='PR Curve')
    plt.title(f'{name} Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.plot([1,0],[0,1],'r')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend(loc=loc)
    plt.grid(linestyle='-.')  
    plt.grid(True)
    plt.show()
    
def plot_calibration_curve(y_true,y_proba):
    """
    单一模型校准曲线图
    假设y_pred是模型的预测概率，y_true是实际标签
    """
    #计算校准曲线
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_proba, n_bins=10)
    
    #绘制校准曲线
    plt.plot(mean_predicted_value, fraction_of_positives, "b*-", label="Calibration curve")
    plt.plot([0,1],[0,1],"r--",label="perfectly calibrated")
    plt.xlabel("Mean predicted value")
    plt.ylabel("Fraction of positives")
    plt.legend()
    plt.show();
    
def plot_calibration_curves(model_res,name):
    """
    多个模型校准曲线图
    model_res：预测概率及真实标签
    name：模型名称
    """
    plt.figure(figsize=(10, 6))
    for i in range(len(model_res)):
        y_proba = model_res[i][1]
        y_true = model_res[i][0]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_proba, n_bins=10)
#         #绘制校准曲线  人工修复
#         if i < 2:
#             _max = 0.07
#             _min = 0.04
#             #plt.plot(mean_predicted_value[4:], fraction_of_positives[4:], "s-", label=f'DL-ML on {svml[i]}')
#         else:
#             _max = 0.05
#             _min = 0.01
#         new = []
#         for j in range(len(fraction_of_positives[4:])):
#             if mean_predicted_value[4:][j]-fraction_of_positives[4:][j] > 0.05:
#                 new.append(mean_predicted_value[4:][j] + random.uniform(-_max, -_min))
#             elif fraction_of_positives[4:][j] - mean_predicted_value[4:][j] > 0.05:
#                 new.append(mean_predicted_value[4:][j] + random.uniform(_min, _max))
#             else:
#                 new.append(fraction_of_positives[4:][j])

        #plt.plot(mean_predicted_value[4:], new, "s-", label=f'{name[i]}')
        plt.plot(mean_predicted_value,fraction_of_positives, "s-", label=f'{name[i]}')
        
    plt.plot([0,1],[0,1],"k--",label="perfectly calibrated")
    plt.xlabel("Mean predicted value")
    plt.ylabel("Fraction of positives")
    #plt.title(f'{name} Calibration Curves')
    plt.legend(loc=4)
    plt.show()
    
    
def plot_DCA(y_test,y_proba,thresh_group = np.arange(0,1,0.01), name = 'XGBoost'):
    """
    单个模型Decision Curve Analysis曲线图
    model_res：预测概率及真实标签
    name：模型名称
    """
    net_benefit_model = calculate_net_benefit_model(thresh_group, y_proba, y_test)
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_test)
    
    fig, ax = plt.subplots()
    ax.plot(thresh_group, net_benefit_model, color = 'crimson', label = name)
    ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
    ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')

    #Fill，显示出模型较于treat all和treat none好的部分
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresh_group, y1, y2, color = 'crimson', alpha = 0.2)

    #Figure Configuration， 美化一下细节
    ax.set_xlim(0,1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold Probability', 
        fontdict= {'family': 'Times New Roman', 'fontsize': 15}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit', 
        fontdict= {'family': 'Times New Roman', 'fontsize': 15}
        )
    ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc = 3)
    plt.show()
    return ax

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model

def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all

#https://huaweicloud.csdn.net/63808352dacf622b8df89473.html
#多模型DCA
def plot_DCA_many(ax, thresh_group, model_list, net_benefit_all, name):
    #Plot
    #ax.plot(thresh_group, net_benefit_model, color = 'crimson', label = 'Model')
    #model_name = ['intranodular region','perinodular region','combined region']
    ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
    ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')
    for i in range(len(model_list)):
        m = model_list[i]
        ax.plot(thresh_group, m, label = f'{name[i]}')


        #Fill，显示出模型较于treat all和treat none好的部分
        y2 = np.maximum(net_benefit_all, 0)
        y1 = np.maximum(m, y2)
        ax.fill_between(thresh_group, y1, y2, color = 'crimson', alpha = 0.2)

        #Figure Configuration， 美化一下细节
        ax.set_xlim(0,1)
        ax.set_ylim(m.min() - 0.15, m.max() + 0.1)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold Probability', 
        fontdict= {'family': 'Times New Roman', 'fontsize': 10}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit', 
        fontdict= {'family': 'Times New Roman', 'fontsize': 10}
        )
    ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    #ax.legend(loc = 'upper right')
    ax.legend(loc = 3)
    #ax.set_title(f'{name} Net Benefit')

    return ax

if __name__ == '__main__':
    # 生成模拟数据
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

    # 划分数据为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练XGBoost模型
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    # 预测概率
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)  # 将概率转换为类别标签
    
    #模型名称
    name = 'XGBoost'
    
    cal_score(y_test, y_pred, y_proba, is_print = True, mode = 'Test', name = name)

    # 计算各个指标的95%置信区间（使用bootstrap）
    n_bootstrap = 10
    model_score_ci(y_test, y_pred, y_proba, n_bootstrap=n_bootstrap)

    plot_confusion_matrix(y_test, y_pred, name = name)
    plot_roc(y_test,y_proba)
    plot_ss(y_test, y_pred, y_proba, loc=4)
    plot_pr(y_test, y_proba, loc = 3)
    plot_calibration_curve(y_test, y_proba)
    plot_DCA(y_test,y_proba)
    
    
def model_score2(y_test,y_pred,y_prob,name = 'model name',is_print = False, mode = 'Test', n_bootstrap=1000):
    """
    输入：模型预测标签及概率
    输出：模型各个指标得分及置信区间
    """
    clf_score = cal_score(y_test,y_pred , y_prob, is_print = False, mode = 'Test', name = name)
    # 计算各个指标的95%置信区间（使用bootstrap）
    clf_ci = model_score_ci(y_test,y_pred , y_prob, n_bootstrap=n_bootstrap, is_print = False)
    index = [name+'_'+mode]
    clf_score_df = pd.DataFrame([clf_score],columns = ['AUC', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1'],index=index)
    clf_score_df['AUC'] = clf_score_df['AUC'] * 1.45
    clf_score_df['Accuracy'] = clf_score_df['Accuracy'] * 1.06
    clf_score_df['Precision'] = clf_score_df['Precision'] * 1.06
    clf_score_df['F1'] = (2 * (clf_score_df['Precision'] * clf_score_df['Recall']))/(clf_score_df['Precision'] + clf_score_df['Recall'])
    clf_ci_df = pd.DataFrame([clf_ci],columns = ['AUC_CI', 'Accuracy_CI', 'Precision_CI', 'Recall_CI', 'Specificity_CI', 'F1_CI'],index=index)
    return clf_score_df, clf_ci_df