"""Implements a classifier for cross-decoding to a different dataset."""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from random import shuffle


class CrossClassifier(object):

    def __init__(self, classifier, test_data, test_labels, **kwargs):
        self.classifier = classifier
        self.kwargs = kwargs
        self.test_data = test_data
        self.test_labels = test_labels

    def fit(self, training_data, training_labels):
        self.classifier.set_params(**self.kwargs)
        return self.classifier.fit(training_data.copy(), training_labels.copy())

    def score(self, fake_test_data, fake_test_labels):
        return self.classifier.score(self.test_data, self.test_labels)

    def get_params(self, deep=True):
        params = self.classifier.get_params(deep)
        params['test_data'] = self.test_data
        params['test_labels'] = self.test_labels
        params['classifier'] = self.classifier
        return params


def permutation_test(clf, X, y, cv, n_permutations):
    X = X.copy()
    y = y.copy()
    score = np.mean(cross_val_score(clf, X, y, cv=cv))
    permutation_scores = np.empty(n_permutations)
    for n in range(n_permutations):
        permutation_scores[n] = np.mean(cross_val_score(clf, X, y, cv=cv))
        np.random.shuffle(X)
    pvalue = np.sum(permutation_scores >= score) / float(n_permutations)
    return score, permutation_scores, pvalue


def toy_data(classes, n_per_class):
    labels = []
    samples = []
    trials = range(n_per_class * 2)
    for c in classes:
        labels += list(c[0]) * n_per_class
        samples += list(np.random.normal(c[1], c[2], n_per_class))
    data = list(zip(labels, samples))
    shuffle(data)
    labels, samples = zip(*data)
    data = list(zip(trials, labels, samples))
    data = pd.DataFrame(data=data, columns=['trial', 'class', 'value'])
    return data
