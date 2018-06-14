# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier

SEED = 0x0803

# 数据降维
def decomposite(data, percentage=100, visualize=False):
    if percentage == 100:
        return data
    else:
        percentage_map = {80: 13, 90: 22}
    pca = PCA(n_components = percentage_map[percentage], svd_solver = 'full', whiten = True)
    pca.fit(data)

    if visualize:
        print(np.sum(pca.explained_variance_ratio_ ))
        print(pca.singular_values_)
        plt.figure(figsize = (8, 5))
        plt.bar(np.arange(108) + 1, pca.singular_values_)
        plt.xticks(np.arange(10) * 10 + 10)
        plt.tight_layout()
        plt.savefig('figures/singular_values.png', dpi = 300)
    return pca.transform(data)

# 评估模型
def evaluate(X_train, X_test, y_train, y_test, base_model='bayes', packing_method=None):
    if base_model == 'bayes':
        model = GaussianNB()
    elif base_model == 'kNN':
        model = KNeighborsClassifier(n_neighbors=13, n_jobs=-1)
    elif base_model == 'dtree':
        model = DecisionTreeClassifier(random_state=SEED)
    elif base_model == 'LR':
        model = LogisticRegression(random_state=SEED)
    elif base_model == 'svm':
        model = SVC(kernel='linear', max_iter=2000000, random_state=SEED)
    elif base_model == 'mlp':
        model = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(20, 20, 20), learning_rate='invscaling', max_iter=100000, random_state=SEED)

    if packing_method == 'bagging':
        model = BaggingClassifier(base_estimator=model, n_estimators=20, random_state=SEED, n_jobs=-1)
    if packing_method == 'boosting':
        model = AdaBoostClassifier(base_estimator=model, n_estimators=20, random_state=SEED, algorithm="SAMME")
    if packing_method == 'random_forest':
        model = RandomForestClassifier(n_estimators=20, random_state=SEED, n_jobs=-1)

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test) * 100
    return accuracy
    
parser = argparse.ArgumentParser(description="ADULT classifiers")
parser.add_argument("--model", type=str, default="dtree")
parser.add_argument("--method", type=str, default="boosting")
parser.add_argument("--pct", type=int, default=80)

if __name__ == '__main__':
    args = parser.parse_args()
    data, label = np.load('adult_data.npy'), np.load('adult_label.npy')
    data = decomposite(data, args.pct)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.2, random_state = SEED)
    print(evaluate(X_train, X_test, y_train, y_test, base_model=args.model, packing_method=args.method))