import numpy as np
import csv
from pathlib import Path

from datasets import get_real_data, get_synthetic_data
from utils import prepare_data
from models import RidgeRegression

RESULTS_DIR = Path("results")

def run_and_save_results(X_train, y_train, X_test, y_test, lambdas, filename, verbose = True):
    n_train = X_train.shape[0]
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    filepath = RESULTS_DIR / filename

    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(['lambda', 'train_error', 'test_error', 'instability'])

        if verbose : print(f"--- Inizio esperimento: {filename} ---")

        for l in lambdas:
            base_model = RidgeRegression(lambda_val=l)
            base_model.fit(X_train, y_train)
            
            # Calcolo del training error
            train_preds = base_model.predict(X_train)
            train_mse = np.mean((train_preds - y_train)**2)
            
            # Calcolo del test error
            base_preds = base_model.predict(X_test)
            test_mse = np.mean((base_preds - y_test)**2)
            
            # Stabilità - LOO
            sum_diffs = 0.0
            for i in range(n_train):
                X_loo = np.delete(X_train, i, axis=0)
                y_loo = np.delete(y_train, i, axis=0)
                
                loo_model = RidgeRegression(lambda_val=l)
                loo_model.fit(X_loo, y_loo)
                loo_preds = loo_model.predict(X_test)
                
                sum_diffs += np.mean(np.abs(base_preds - loo_preds))
            
            avg_instability = sum_diffs / n_train
            
            writer.writerow([l, train_mse, test_mse, avg_instability])
            if verbose : print(f"Lambda: {l:.4f} completato.")

    if verbose : print(f"--- Risultati salvati in: {filepath} ---\n")

if __name__ == "__main__":
    lambdas = np.logspace(-3, 5, 30)

    # Inserito lo 0.0 all'inizio (Unregularized Least Squares)
    lambdas = np.insert(lambdas, 0, 0.0)

    # EXP 1: Dataset Sintetico
    X_s, y_s = get_synthetic_data(n_samples=60, n_features=50, noise=20.0)
    Xs_train, Xs_test, ys_train, ys_test = prepare_data(X_s, y_s)
    run_and_save_results(Xs_train, ys_train, Xs_test, ys_test, lambdas, "results_synthetic.csv")

    # EXP 2: Dataset Reale (Diabetes)
    X_r, y_r = get_real_data()
    Xr_train, Xr_test, yr_train, yr_test = prepare_data(X_r, y_r)
    run_and_save_results(Xr_train, yr_train, Xr_test, yr_test, lambdas, "results_real.csv")