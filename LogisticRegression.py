import numpy as np

class LogisticRegression():
    def __init__(self, learning_rate=0.00015, num_iterations=2000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_predictions = np.dot(X, self.weights) + self.bias
            logistic_predictions = self.sigmoid(linear_predictions)

            dw = (1 / num_samples) * np.dot(X.T, (logistic_predictions - y))
            db = (1 / num_samples) * np.sum(logistic_predictions - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
        

    def predict(self, X):
        linear_prediction = np.dot(X, self.weights) + self.bias
        y_prediction = self.sigmoid(linear_prediction)

        return [0 if y <= 0.5 else 1 for y in y_prediction]
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))