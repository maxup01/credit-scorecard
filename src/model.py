import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate: float = 1.0, num_iterations: int = 1000):
        """Set up an untrained logistic regression model.

        `weights` and `bias` are left as placeholders here since the number
        of features isn't known yet — `initialize_weights`
        replaces `weights` with a properly sized zero vector
        once the training data's shape is known.

        Args:
            learning_rate: Step size for each gradient descent update — how
                far `weights`/`bias` move against the gradient on every
                iteration. Too high can overshoot and fail to converge; too
                low converges very slowly.
            num_iterations: Number of gradient descent passes to run over
                the full training set during `train_model`.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights: np.ndarray = np.ndarray(shape=(0, 0))
        self.bias = 0

    def initialize_weights(self, dimensions: int):
        self.weights = np.zeros((dimensions, 1))
        self.bias = 0

    def hypothesis(self, X: np.ndarray) -> np.ndarray:
        dot_product = np.dot(self.weights.T, X)

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

        params = {"weights": self.weights, "bias": self.bias}

        gradients, cost = self.gradient_position(X, Y)

        return params, gradients, costs

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X)
        m = X.shape[1]

        Y_prediction = np.zeros((1, m))

        H = self.hypothesis(X)

        for i in range(H.shape[1]):
            if H[0, i] >= 0.5:
                Y_prediction[0, i] = 1
            else:
                Y_prediction[0, i] = 0

        return Y_prediction

    def train_model(
            self,
            X_train: np.ndarray,
            Y_train: np.ndarray,
            X_test: np.ndarray,
            Y_test: np.ndarray
    ) -> dict[str, np.ndarray | list[float] | float]:
        dimensions = np.shape(X_train)[0]
        self.initialize_weights(dimensions)

        (params, gradients, costs) = self.gradient_descent(X_train, Y_train)

        self.weights = params["weights"]
        self.bias = params["bias"]

        Y_prediction_test = self.predict(X_test)
        Y_prediction_train = self.predict(X_train)

        train_score = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
        test_score = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100

        print(f"training accuracy: { 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100 }.")
        print(f"test accuracy: { 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100 }.")

        return {
            "costs": costs,
            "Y_prediction_test": Y_prediction_test,
            "Y_prediction_train": Y_prediction_train,
            "weights": self.weights,
            "bias": self.bias,
            "learning_rate": self.learning_rate,
            "train_accuracy": train_score,
            "test_accuracy": test_score
        }

    @staticmethod
    def calculate_gradient(H: np.ndarray, X: np.ndarray, Y: np.ndarray) -> dict[str, np.ndarray | float]:
        m = X.shape[1]

        dw = np.dot(X, (H - Y).T) / m
        db = np.sum(H - Y) / m

        return {"dw": dw, "db": db}

    @staticmethod
    def cost(H: np.ndarray, Y: np.ndarray) -> float:
        cost = -np.sum(Y * np.log(H) + (1 - Y) * np.log(1 - H)) / H.shape[1]
        cost = np.squeeze(cost)
        return cost

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))