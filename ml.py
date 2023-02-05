import numpy as np

class MultipleLinearRegression:
    def __init__(self):
        self.weights = None
    
    def fit(self, X, y):
        # Add a column of ones to X to account for the bias term
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)

        # Calculate the weights using the normal equation
        X_transpose = np.transpose(X)
        theta = np.linalg.inv(X_transpose @ X) @ X_transpose @ y
        self.weights = theta
    
    def predict(self, X):
        # Add a column of ones to X to account for the bias term
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)

        # Calculate the predictions using the weights
        y_pred = X @ self.weights
        return y_pred
