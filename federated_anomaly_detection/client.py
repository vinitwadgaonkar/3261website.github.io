import numpy as np
from .model import AnomalyModel

class FederatedClient:
    def __init__(self, data, labels, noise_std=0.1):
        self.model = AnomalyModel()
        self.data = data
        self.labels = labels
        self.noise_std = noise_std

    def train(self):
        self.model.fit(self.data, self.labels)

    def get_noisy_weights(self):
        weights = self.model.get_weights()
        noise = np.random.normal(scale=self.noise_std, size=weights.shape)
        return weights + noise
