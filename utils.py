import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = X.shape[0]
    
    indices = np.arange(n_samples)
    
    np.random.shuffle(indices)
    
    split_idx = int(n_samples * (1 - test_size))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def prepare_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)
    epsilon = 1e-8 

    X_train_scaled = (X_train - X_train_mean) / (X_train_std + epsilon)
    X_test_scaled = (X_test - X_train_mean) / (X_train_std + epsilon)

    y_train_mean = np.mean(y_train)
    y_train_centered = y_train - y_train_mean
    y_test_centered = y_test - y_train_mean

    return X_train_scaled, X_test_scaled, y_train_centered, y_test_centered

def custom_make_regression(n_samples=200, n_features=20, noise=20.0, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
        
    # Input X campionati da una distribuzione normale standard (Media 0, Varianza 1)
    X = np.random.randn(n_samples, n_features)
    
    # Vettore dei pesi estratto casualmente, ad esempio da una distribuzione uniforme tra -10 e 10
    true_weights = np.random.uniform(low=-10.0, high=10.0, size=n_features)
    
    y = X @ true_weights
    
    # Rumore Gaussiano per studiare la robustezza
    if noise > 0.0:
        gaussian_noise = np.random.normal(loc=0.0, scale=noise, size=n_samples)
        y += gaussian_noise
        
    return X, y
