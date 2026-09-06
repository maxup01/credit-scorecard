import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate: float = 1.0, num_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = []
        self.bias = 0

    def initialize_weights(self, dimensions: int):
        self.weights = np.zeros((dimensions, 1))
        self.bias = 0

    def hypothesis(self, X: np.ndarray) -> np.ndarray:
        dot_product = np.dot(X, self.weights)

        return sigmoid(dot_product + self.bias)

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))