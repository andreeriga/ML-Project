import numpy as np
import csv
from pathlib import Path

from utils import custom_make_regression, prepare_data
from models import RidgeRegression

RESULTS_DIR = Path("results")

def run_size_experiment(verbose = False):
    # Tre dimensioni del dataset
    dataset_sizes = [40, 150, 500]
    n_features = 20
    noise_level = 25.0
    
    lambdas = np.logspace(-3, 3, 20)
    lambdas = np.insert(lambdas, 0, 0.0)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_DIR / "results_dataset_size.csv"
    
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['N', 'lambda', 'test_error', 'instability'])

        for N in dataset_sizes:
            if verbose : print(f"\n--- Inizio esperimento per N = {N} ---")
            
            X, y = custom_make_regression(n_samples=N, n_features=n_features, noise=noise_level)
            X_train, X_test, y_train, y_test = prepare_data(X, y)
            
            n_train = X_train.shape[0]

            for l in lambdas:
                base_model = RidgeRegression(lambda_val=l)
                base_model.fit(X_train, y_train)
                base_preds = base_model.predict(X_test)
                test_mse = np.mean((base_preds - y_test)**2)
                
                sum_diffs = 0.0
                for i in range(n_train):
                    X_loo = np.delete(X_train, i, axis=0)
                    y_loo = np.delete(y_train, i, axis=0)
                    
                    loo_model = RidgeRegression(lambda_val=l)
                    loo_model.fit(X_loo, y_loo)
                    loo_preds = loo_model.predict(X_test)
                    
                    sum_diffs += np.mean(np.abs(base_preds - loo_preds))
                
                avg_instability = sum_diffs / n_train
                
                writer.writerow([N, l, test_mse, avg_instability])
                if verbose :print(f"N={N:3d} | Lambda: {l:7.3f} | Instability: {avg_instability:.4f}")

    if verbose : print(f"\n--- Esperimento Dimensioni concluso. Salvato in {filepath} ---")

if __name__ == "__main__":
    run_size_experiment()