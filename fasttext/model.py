# fastText Model representation in Python
import numpy as np
from numpy.linalg import norm

# Class for classifier model
class SupervisedModel(object):
    def __init__(self, model, labels, label_prefix, encoding='utf-8'):
        self._model = model
        self.labels = labels
        self.dim = model.dim
        self.ws = model.ws
        self.epoch = model.epoch
        self.min_count = model.minCount
        self.neg = model.neg
        self.word_ngrams = model.wordNgrams
        self.loss_name = model.lossName.decode(encoding)
        self.model_name = model.modelName.decode(encoding)
        self.bucket = model.bucket
        self.minn = model.minn
        self.maxn = model.maxn
        self.lr_update_rate = model.lrUpdateRate
        self.t = model.t
        self.label_prefix = label_prefix
        self.encoding = encoding

    def test(self, test_file, k=1):
        return self._model.classifier_test(test_file, k, self.encoding)

    def predict(self, texts, k=1):
        all_labels = []
        for text in texts:
            if text[-1] != '\n':
                text += '\n'
            labels = self._model.classifier_predict(text, k,
                    self.label_prefix, self.encoding)
            all_labels.append(labels)
        return all_labels

    def predict_proba(self, texts, k=1):
        results = []
        for text in texts:
            if text[-1] != '\n':
                text += '\n'
            result = self._model.classifier_predict_prob(text, k,
                    self.label_prefix, self.encoding)
            results.append(result)
        return results