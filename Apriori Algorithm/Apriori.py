from __future__ import division, print_function
import argparse
import numpy as np
import time
from itertools import combinations, chain

parser = argparse.ArgumentParser(description='Apriori')
parser.add_argument('--data-size', type=int, default=20)
parser.add_argument('--classes', type=int, default=5)
parser.add_argument('--min-support', type=float, default=0.18)
parser.add_argument('--min-confidence', type=float, default=0.7)
parser.add_argument('--seed', type=float, default=233)
parser.add_argument('--verbose', type=bool, default=False)
parser.add_argument('--generate-dataset', type=bool, default=False)
args = parser.parse_args()

def rand_cov(classes):
    cov = np.eye(classes)
    for i, j in combinations(range(classes), 2):
        cov[i, j] = cov[j, i] = np.random.uniform(-1, 1)
    return cov

def format_data(raw_data):
    idx = np.argwhere(raw_data > 0)
    return [frozenset(idx[idx[:, 0] == data_id, 1]) for data_id in range(raw_data.shape[0])]

def generate_dataset(data_size, classes, seed=233):
    np.random.seed(seed)
    cov = rand_cov(classes)
    raw_data = np.random.multivariate_normal(np.zeros(classes), cov, data_size)
    return format_data(raw_data)

def load_dataset(fname):
    data = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            data.append(eval(line[:-1]))
    return data

def count(item, data):
    return sum([1 if item.issubset(d) else 0 for d in data])

def valid(x, itemset):
    return sum([int(x.difference(frozenset([key])) in itemset) for key in x]) == len(x)

def generate_next_itemset(itemset):
    next_itemset = [a.union(b) for a, b in combinations(itemset, 2) if len(a.union(b)) - len(a) == 1]
    next_itemset = frozenset(filter(lambda x: valid(x, itemset), next_itemset))
    return next_itemset

def generate_rules(x, data, min_confidence):
    cnt = count(x, data)
    rule_set = [(x.difference(frozenset([item])), frozenset([item])) for item in x]
    rule_set = frozenset(filter(lambda t: count(t[1], data) > min_confidence * cnt, rule_set))
    rules = []
    while len(rule_set) > 0:
        rules.append(rule_set)
        rule_set = frozenset(chain(*[[(rule[0].difference(frozenset([item])), rule[1].union(frozenset([item]))) for item in rule[0]] for rule in rule_set]))
        rule_set = frozenset(filter(lambda t: cnt > min_confidence * count(t[0], data), rule_set))
    return frozenset(chain(*rules))

def Apriori(data, min_support, min_confidence, verbose=False):
    itemset = frozenset(map(lambda x: frozenset([x]), chain(*data)))
    freqset = []
    # 找出频繁项集
    while len(itemset) > 0:
        itemset = frozenset(filter(lambda x: count(x, data) > min_support * len(data), itemset))
        freqset.append(itemset)
        itemset = generate_next_itemset(itemset)
    # 生成关联规则
    rules = frozenset(map(lambda x: generate_rules(x, data, min_confidence), freqset[-1]))

    freqset = list(chain(*freqset[1-int(verbose):]))
    rules = frozenset(filter(lambda x: len(x[0]) > 0, chain(*rules)))
    return freqset, rules

data = generate_dataset(args.data_size, args.classes, args.seed) if args.generate_dataset else load_dataset('GroceryStoreDataSet.txt')
t = time.time()
freqset, rules = Apriori(data, args.min_support, args.min_confidence, args.verbose)
print('%0.6f' % (time.time() - t))
print(freqset)
print(rules)
