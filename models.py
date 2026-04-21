import numpy as np

class RidgeRegression:
    def __init__(self, lambda_val=0.0):
        # Con lambda_val = 0.0, Ridge Regression si riduce a Unregularized Least Squares
        self.lambda_val = lambda_val
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        X_b = np.column_stack((np.ones(n_samples), X))
        
        I = np.eye(n_features + 1)
        
        # Non regolarizzo il bias
        I[0, 0] = 0 
        
        A = X_b.T @ X_b + self.lambda_val * I
        b = X_b.T @ y
        
        self.weights = np.linalg.solve(A, b)

    def predict(self, X):
        n_samples = X.shape[0]
        X_b = np.column_stack((np.ones(n_samples), X))
        
        return X_b @ self.weights