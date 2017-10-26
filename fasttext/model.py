# fastText Model representation in Python
import numpy as np
from numpy.linalg import norm

# Class for classifier model
class SupervisedModel(object):
    def __init__(self, model, label_prefix, encoding='utf-8'):
        self._model = model
        self.label_prefix = label_prefix
        self.encoding = encoding

    def predict_proba(self, texts):
        results = []
        for text in texts:
            if text[-1] != '\n':
                text += '\n'
            result = self._model.classifier_predict_prob(
                text,
                self.label_prefix,
                self.encoding
            )
            results.append(result)
        return results