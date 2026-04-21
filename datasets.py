import numpy as np
from sklearn.datasets import load_diabetes
from utils import custom_make_regression

def get_synthetic_data(n_samples=200, n_features=20, noise=20.0):
    X, y = custom_make_regression(
        n_samples=n_samples, 
        n_features=n_features, 
        noise=noise, 
        random_state=42
    )
    return X, y

def get_real_data():
    dataset = load_diabetes()
    return dataset.data, dataset.target