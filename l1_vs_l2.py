from models import RidgeRegression, LassoRegression
from utils import custom_make_regression, prepare_data
import numpy as np
import csv
from pathlib import Path

RESULTS_DIR = Path("results")

def run_and_save_l1_vs_l2_results(X_train, y_train, X_test, y_test, lambdas, filename, verbose=True):
    n_train = X_train.shape[0]
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    filepath = RESULTS_DIR / filename

    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow([
            'lambda', 
            'ridge_train_error', 'ridge_test_error', 'ridge_instability', 'ridge_zeros',
            'lasso_train_error', 'lasso_test_error', 'lasso_instability', 'lasso_zeros'
        ])

        if verbose: print(f"--- Inizio esperimento L1 vs L2: {filename} ---")
        if verbose: print("Nota: Lasso usa Coordinate Descent, potrebbe impiegare qualche secondo in più...")

        for l in lambdas:

            ridge_base = RidgeRegression(lambda_val=l)
            ridge_base.fit(X_train, y_train)
            
            ridge_train_preds = ridge_base.predict(X_train)
            ridge_train_mse = np.mean((ridge_train_preds - y_train)**2)
            
            ridge_test_preds = ridge_base.predict(X_test)
            ridge_test_mse = np.mean((ridge_test_preds - y_test)**2)
            
            ridge_zeros = np.sum(np.abs(ridge_base.weights[1:]) == 0.0)
            
            # Stabilità - LOO
            ridge_sum_diffs = 0.0
            for i in range(n_train):
                X_loo = np.delete(X_train, i, axis=0)
                y_loo = np.delete(y_train, i, axis=0)
                
                ridge_loo = RidgeRegression(lambda_val=l)
                ridge_loo.fit(X_loo, y_loo)
                ridge_loo_preds = ridge_loo.predict(X_test)
                
                ridge_sum_diffs += np.mean(np.abs(ridge_test_preds - ridge_loo_preds))
            
            ridge_instability = ridge_sum_diffs / n_train

            lasso_base = LassoRegression(lambda_val=l)
            lasso_base.fit(X_train, y_train)
            
            lasso_train_preds = lasso_base.predict(X_train)
            lasso_train_mse = np.mean((lasso_train_preds - y_train)**2)
            
            lasso_test_preds = lasso_base.predict(X_test)
            lasso_test_mse = np.mean((lasso_test_preds - y_test)**2)
            
            lasso_zeros = np.sum(np.abs(lasso_base.weights[1:]) == 0.0)
            
            # Stabilità - LOO
            lasso_sum_diffs = 0.0
            for i in range(n_train):
                X_loo = np.delete(X_train, i, axis=0)
                y_loo = np.delete(y_train, i, axis=0)
                
                lasso_loo = LassoRegression(lambda_val=l)
                lasso_loo.fit(X_loo, y_loo)
                lasso_loo_preds = lasso_loo.predict(X_test)
                
                lasso_sum_diffs += np.mean(np.abs(lasso_test_preds - lasso_loo_preds))
            
            lasso_instability = lasso_sum_diffs / n_train

            writer.writerow([
                l, 
                ridge_train_mse, ridge_test_mse, ridge_instability, ridge_zeros,
                lasso_train_mse, lasso_test_mse, lasso_instability, lasso_zeros
            ])
            
            if verbose: 
                print(f"Lambda: {l:7.4f} completato. | Zeri azzerati -> Ridge: {ridge_zeros:2d}, Lasso: {lasso_zeros:2d}")

    if verbose: print(f"--- Risultati salvati in: {filepath} ---\n")

if __name__ == "__main__":
    # Dataset con molte feature per far lavorare bene la Lasso
    X_l1l2, y_l1l2 = custom_make_regression(n_samples=100, n_features=40, noise=25.0)
    X_train_l1l2, X_test_l1l2, y_train_l1l2, y_test_l1l2 = prepare_data(X_l1l2, y_l1l2)

    lambdas = np.logspace(-3, 3, 20)
    lambdas = np.insert(lambdas, 0, 0.0)

    run_and_save_l1_vs_l2_results(
        X_train_l1l2, y_train_l1l2, 
        X_test_l1l2, y_test_l1l2, 
        lambdas, 
        filename="results_l1_vs_l2.csv", 
        verbose=True
    )