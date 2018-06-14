# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing

NORM = True

# 数据预处理
def encode(df):
    result = np.zeros((16281, 108))
    idx = 0
    for column in df.columns:
        if column != '收入':
            if df.dtypes[column] == np.object:
                tmp = preprocessing.OneHotEncoder(sparse=False).fit_transform(preprocessing.LabelEncoder().fit_transform(df[column]).reshape(-1, 1))
                result[:, idx: idx+tmp.shape[1]] = tmp
                idx += tmp.shape[1]
            else:
                result[:, idx:idx+1] = df[column].values.reshape(-1, 1)
                idx += 1
    return result

# 数据标准化
def normalize(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    sigma[sigma < 1] = 1
    return (x - mu) / sigma

headers = ['年龄', '工作类别', '最终权重', '教育程度', '受教育时间', '婚姻状况', '职业', '家庭关系', '种族', '性别', '资本收益', '资本亏损', '每周工作小时数', '祖国', '收入']
df = pd.read_csv('adult.txt', names = headers)

result = encode(df)
if NORM:
    result = normalize(result)
np.save('adult_data.npy', result)
np.save('adult_label.npy', df['收入'].values.reshape(-1))