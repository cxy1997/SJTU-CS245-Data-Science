# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
np.set_printoptions(formatter = {'float_kind': lambda x: "%.4f" % x})
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset_file = 'boston.txt'

def preprocessing(fname):
    with open(fname, 'r') as f:
        document = f.readlines()
    with open(fname, 'w') as f:
        for line in document:
            f.write(' '.join(list(filter(lambda x: x != '', line[:-1].split(' ')))) + '\n')

def show_property(df, key):
    print('%s 最大值：' % key, df[key].max())
    print('%s 最小值：' % key, df[key].min())
    print('%s 均值：' % key, df[key].mean())
    print('%s 中位数：' % key, df[key].median())
    print('%s 标准差：' % key, df[key].std())
    # print(' & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f' % (df[key].max(), df[key].min(), df[key].mean(), df[key].median(), df[key].std()))
    plt.figure(figsize = (8, 5))
    ax = df[key].plot(kind = 'hist')
    ax.set(xlabel = key, ylabel = "Value")
    plt.tight_layout()
    plt.savefig('figures/%s.png' % key, dpi = 300)

def plot_singular_values(df):
    pca = PCA(n_components = 14, svd_solver = 'full', whiten = True)
    pca.fit(df)
    print(pca.singular_values_)

    plt.figure(figsize = (8, 5))
    plt.bar(np.arange(14) + 1, pca.singular_values_)
    plt.xticks(np.arange(14) + 1)
    plt.tight_layout()
    plt.savefig('figures/singular_values.png', dpi = 300)

def normalize(df, key):
    df[key] = (df[key] - df[key].mean()) / df[key].std()

def plot_transform(df, n_components = 1):
    pca = PCA(n_components = n_components, svd_solver = 'full', whiten = True)
    new_data = pca.fit_transform(df)
    lr = LinearRegression()
    lr.fit(new_data, df['MEDV'])
    print(lr.coef_, lr.intercept_)
    print(lr.score(new_data, df['MEDV']))

    if n_components == 1:
        plt.figure(figsize = (8, 5))
        plt.scatter(new_data, df['MEDV'])
        plt.plot(new_data, lr.predict(new_data), 'r')
        plt.tight_layout()
        plt.savefig('figures/cmp.png', dpi = 300)
    return lr.score(new_data, df['MEDV'])

def plot_scores(scores):
    print(scores)
    
    plt.figure(figsize = (8, 5))
    plt.plot(np.arange(14) + 1, scores)
    plt.xticks(np.arange(14) + 1)
    plt.tight_layout()
    plt.savefig('figures/scores.png', dpi = 300)

if __name__ == '__main__':
    preprocessing(dataset_file)
    df = pd.read_csv(dataset_file, sep = ' ')
    for key in df.keys():
        show_property(df, key)
        normalize(df, key)
    plot_singular_values(df)
    scores = [plot_transform(df, n_components = i) for i in range(1, 15)]
    plot_scores(scores)