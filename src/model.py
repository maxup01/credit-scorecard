import numpy as np
from prompt_toolkit.key_binding.bindings.named_commands import self_insert


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

        return self.sigmoid(dot_product + self.bias)

    def gradient_position(self, X: np.ndarray, Y: np.ndarray) -> tuple[dict[str, np.ndarray | float], float]:
        H = self.hypothesis(X)

        cost = self.cost(H, Y)
        grads = self.calculate_gradient(H, X, Y)

        return grads, cost

    def gradient_descent(self, X: np.ndarray, Y: np.ndarray):
        costs = []

        for i in range(0, self.num_iterations):
            gradients, cost = self.gradient_position(X, Y)

            dw = gradients["dw"]
            db = gradients["db"]

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 100 == 0:
                costs.append(cost)

                print(f"Cost after iteration {i}: {cost}")

        params = {"weights": self.weights, "b": self.bias}

        gradients, cost = self.gradient_position(X, Y)

        return params, gradients, costs

    @staticmethod
    def calculate_gradient(H: np.ndarray, X: np.ndarray, Y: np.ndarray) -> dict[str, np.ndarray | float]:
        m = X.shape[1]

        dw = np.dot(X, (H - Y).T) / m
        db = np.sum(H - Y) / m

        return {"dw": dw, "db": db}

    @staticmethod
    def cost(H: np.ndarray, Y: np.ndarray) -> float:
        cost = -np.sum(Y * np.log(H) + (1 - Y) * np.log(1 - H)) / len(H)
        cost = np.squeeze(cost)
        return cost

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))