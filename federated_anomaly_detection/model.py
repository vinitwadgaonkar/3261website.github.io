import numpy as np
from sklearn.linear_model import SGDClassifier

class AnomalyModel:
    """Simple logistic regression model for anomaly detection."""

    def __init__(self):
        self.clf = SGDClassifier(loss="log_loss", max_iter=1, learning_rate="constant", eta0=0.01, warm_start=True)
        self.is_fitted = False

    def fit(self, X, y):
        self.clf.partial_fit(X, y, classes=np.array([0, 1]))
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model has not been fitted")
        return (self.clf.predict_proba(X)[:, 1] > 0.5).astype(int)

    def get_weights(self):
        return np.concatenate([self.clf.coef_.flatten(), self.clf.intercept_])

    def set_weights(self, weights):
        coef = weights[:-1].reshape(1, -1)
        intercept = weights[-1:]
        self.clf.coef_ = coef
        self.clf.intercept_ = intercept
        self.is_fitted = True
