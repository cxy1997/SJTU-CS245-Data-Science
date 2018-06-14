from __future__ import division, print_function
import time
import numpy as np
import collections
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def visualize(digits):
    print(digits.keys())
    print(digits['data'].shape)
    print(digits['target'].shape)
    print(collections.Counter(digits['target']))
    print(digits['images'].shape)
    plt.figure(figsize=(3, 3))
    plt.imshow(digits['images'][-1], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.savefig('figures/dataset_demo.png', dpi = 300)

def plot_curve(accuracy, fname):
    plt.figure(figsize = (8, 5))
    plt.plot(range(1, 21), accuracy)
    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy / %')
    plt.tight_layout()
    plt.savefig('figures/%spng' % fname, dpi = 300)

def plot_confusion_matrix(conf_arr, fname):
    norm_conf = conf_arr / np.sum(conf_arr, axis = 0)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(norm_conf, cmap=plt.cm.jet, interpolation='nearest')

    for x in range(10):
        for y in range(10):
            ax.annotate(str(conf_arr[x, y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')

    plt.xticks(range(10), range(10))
    plt.yticks(range(10), range(10))
    plt.savefig('figures/%s.png' % fname, dpi = 300)

def singleLR(digits):
    LR = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(digits['data'], digits['target'], test_size = 0.2, random_state = 817)
    begin_time = time.time()
    LR.fit(X_train, y_train)
    print('Training takes %0.2f seconds.' % (time.time() - begin_time))
    begin_time = time.time()
    y_pred = LR.predict(X_test)
    print('Predicting takes %0.3f seconds.' % (time.time() - begin_time))
    conf_arr = confusion_matrix(y_test, y_pred, labels = range(10))
    print('Prediction accuracy: %0.2f' % (LR.score(X_test, y_test) * 100))
    plot_confusion_matrix(conf_arr, 'confusion_matrix')

def baggingLR(digits, n_estimators = 10):
    BLR = BaggingClassifier(base_estimator = LogisticRegression(), n_estimators = n_estimators, random_state = 817)
    X_train, X_test, y_train, y_test = train_test_split(digits['data'], digits['target'], test_size = 0.2, random_state = 817)
    begin_time = time.time()
    BLR.fit(X_train, y_train)
    print('Training takes %0.2f seconds.' % (time.time() - begin_time))
    begin_time = time.time()
    y_pred = BLR.predict(X_test)
    print('Predicting takes %0.3f seconds.' % (time.time() - begin_time))
    accuracy = BLR.score(X_test, y_test) * 100
    print('Prediction accuracy: %0.2f' %  accuracy)
    return accuracy
    conf_arr = confusion_matrix(y_test, y_pred, labels = range(10))
    plot_confusion_matrix(conf_arr, 'confusion_matrix_bagging')

def boostingLR(digits, n_estimators = 10):
    BLR = AdaBoostClassifier(base_estimator = LogisticRegression(), n_estimators = n_estimators, random_state = 817)
    X_train, X_test, y_train, y_test = train_test_split(digits['data'], digits['target'], test_size = 0.2, random_state = 817)
    begin_time = time.time()
    BLR.fit(X_train, y_train)
    print('Training takes %0.2f seconds.' % (time.time() - begin_time))
    begin_time = time.time()
    y_pred = BLR.predict(X_test)
    print('Predicting takes %0.3f seconds.' % (time.time() - begin_time))
    accuracy = BLR.score(X_test, y_test) * 100
    print('Prediction accuracy: %0.2f' %  accuracy)
    return accuracy
    conf_arr = confusion_matrix(y_test, y_pred, labels = range(10))
    plot_confusion_matrix(conf_arr, 'confusion_matrix_bagging')

np.random.seed(817)
digits = datasets.load_digits()
visualize(digits)
singleLR(digits)
accuracy = [baggingLR(digits, i) for i in range(1, 21)]
plot_curve(accuracy, 'bagging_accuracy')
accuracy = [boostingLR(digits, i) for i in range(1, 21)]
plot_curve(accuracy, 'boosting_accuracy')