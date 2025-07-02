import numpy as np
from sklearn.datasets import make_classification
from .client import FederatedClient
from .server import Aggregator


def simulate(num_clients=10, samples_per_client=100):
    X, y = make_classification(n_samples=num_clients * samples_per_client,
                               n_features=20, n_informative=15,
                               n_redundant=5, random_state=42)
    client_weights = []
    clients = []
    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client
        c = FederatedClient(X[start:end], y[start:end])
        c.train()
        client_weights.append(c.get_noisy_weights())
        clients.append(c)

    agg = Aggregator(n_features=X.shape[1])
    agg.aggregate(client_weights)

    accuracy = agg.evaluate(X, y)
    print(f"Aggregated model accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    simulate()
