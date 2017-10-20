"""Implements a classifier for cross-decoding to a different dataset."""


class CrossClassifier(object):

    def __init__(self, classifier, test_data, test_labels, **kwargs):
        self.classifier = classifier
        self.classifier.set_params(**kwargs)
        self.test_data = test_data
        self.test_labels = test_labels

    def fit(self, training_data, training_labels):
        return self.classifier.fit(training_data, training_labels)

    def score(self, fake_test_data, fake_test_labels):
        return self.classifier.score(self.test_data, self.test_labels)

    def get_params(self, deep=True):
        params = self.classifier.get_params(deep)
        params['test_data'] = self.test_data
        params['test_labels'] = self.test_labels
        params['classifier'] = self.classifier
        return params

    def predict(self, test_data):
        return self.classifier.predict(test_data)
