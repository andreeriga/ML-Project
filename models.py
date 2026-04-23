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
class LassoRegression:
    def __init__(self, lambda_val=1.0, max_iter=1000, tol=1e-4):
        self.lambda_val = lambda_val
        self.max_iter = max_iter
        # Tolleranza per fermare il ciclo quando i pesi non cambiano significativamente
        self.tol = tol 
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Bias
        X_b = np.column_stack((np.ones(n_samples), X))
        
        self.weights = np.zeros(n_features + 1)
        
        z = np.sum(X_b**2, axis=0)
        
        for iteration in range(self.max_iter):
            weights_old = self.weights.copy()
            
            for j in range(n_features + 1):
                # Se una feature è costante viene ignorata
                if z[j] == 0:
                    continue
                
                predictions = X_b @ self.weights
                residual = y - predictions
                
                rho_j = X_b[:, j].T @ (residual + X_b[:, j] * self.weights[j])
                
                if j == 0:
                    # Bias non regolarizzato
                    self.weights[j] = rho_j / z[j]
                else :
                    if rho_j < -self.lambda_val:
                        self.weights[j] = (rho_j + self.lambda_val) / z[j]
                    elif rho_j > self.lambda_val:
                        self.weights[j] = (rho_j - self.lambda_val) / z[j]
                    else:
                        # Peso a zero se rho è tra -lambda e lambda
                        self.weights[j] = 0.0
                        
            # Controllo della convergenza
            if np.max(np.abs(self.weights - weights_old)) < self.tol:
                break

    def predict(self, X):
        n_samples = X.shape[0]
        X_b = np.column_stack((np.ones(n_samples), X))
        return X_b @ self.weights