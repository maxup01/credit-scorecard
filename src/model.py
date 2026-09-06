class LogisticRegression:
    def __init__(self, learning_rate: float = 1.0, num_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = []
        self.bias = 0