import numpy as np
from .model import AnomalyModel

class Aggregator:
    def __init__(self, n_features):
        self.global_model = AnomalyModel()
        # initialize weights to zeros
        self.global_model.clf.coef_ = np.zeros((1, n_features))
        self.global_model.clf.intercept_ = np.zeros(1)
        self.global_model.is_fitted = True

    def aggregate(self, client_weights):
        avg_weights = np.mean(client_weights, axis=0)
        self.global_model.set_weights(avg_weights)

    def evaluate(self, X, y):
        preds = self.global_model.predict(X)
        return np.mean(preds == y)
