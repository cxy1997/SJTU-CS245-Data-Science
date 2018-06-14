from __future__ import division, print_function
from itertools import combinations
import os
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

def show_basic_info(dataset):
    print(dataset['DESCR'])
    print(dataset['data'].shape)
    print(dataset['target'].shape)
    print(dataset['target_names'])
    print(dataset['feature_names'])

def stat(feature):
    print('最小值：', np.min(feature))
    print('最大值：', np.max(feature))
    print('均值：', np.mean(feature))
    print('中位数：', np.median(feature))
    print('标准差：', np.std(feature))

def boxplot(data, labels):
    plt.figure(figsize = (8, 5))
    ax = plt.subplot()
    bp = ax.boxplot([data[:, i] for i in range (4)], notch = True, meanline = True, patch_artist = True)
    for whisker in bp['whiskers']:
        whisker.set(color = 'y', linewidth = 2)
    for cap in bp['caps']:
        cap.set(color = '#7570b3', linewidth = 2)
    for median in bp['medians']:
        median.set(color = 'r', linewidth = 2)
    for flier in bp['fliers']:
        flier.set(marker = 'o', color = 'k', alpha = 0.5)
    ax.set_xticks(range(5))
    ax.set(xlabel = "Features", ylabel = "Value/cm", xlim = (0.5, 4.5))
    ax.set_xticklabels([''] + labels)
    plt.grid(axis = 'y')
    if not os.path.isdir('figures'):
        os.mkdir('figures')
    plt.savefig('figures/boxplot.png', dpi = 300)

def corr(variables):
    n = len(variables)
    result = np.ones((n, n))
    for i, j in combinations(range(n), 2):
        result[i, j] = result[j, i] = np.corrcoef(variables[i], variables[j])[0, 1]
    return result

iris = datasets.load_iris()

show_basic_info(iris)

for i in range(4):
    print(iris['feature_names'][i])
    stat(iris['data'][:, i])
boxplot(iris['data'], iris['feature_names'])

print(corr([iris['data'][:, i] for i in range(4)] + [iris['target']]))