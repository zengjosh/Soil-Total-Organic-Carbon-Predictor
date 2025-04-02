"""
improved_xgboost.py

This script demonstrates:
1. Stratified train-test split based on OC bins.
2. Log transform of the OC target to handle heavy skew.
3. Weighted training for the highest OC bin.
4. Hyperparameter tuning with GridSearchCV.
5. Evaluation of performance overall and by OC bin.

Author: [Your Name]
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    # --------------------------------------------------
    # 1) LOAD AND PREPROCESS DATA
    # --------------------------------------------------
    df = pd.read_csv('LUCAS-SOIL-2018.csv')

    # Relevant columns
    cols = ['pH_H2O', 'EC', 'P', 'N', 'K', 'Elev', 'OC']
    df = df[cols].copy()

    # Replace "< LOD" for P, N, K
    lod_dict = {'P': 10, 'N': 0.2, 'K': 10}
    for col, lod in lod_dict.items():
        df[col] = df[col].replace(r'\s*<\s*LOD\s*', lod/2, regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert other columns to numeric
    for col in ['pH_H2O', 'EC', 'Elev', 'OC']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop missing
    df.dropna(inplace=True)

    # Reset index so stratified indices align
    df.reset_index(drop=True, inplace=True)

    # --------------------------------------------------
    # 2) STRATIFIED SPLIT (BASED ON OC QUARTILES)
    # --------------------------------------------------
    df['OC_bin'] = pd.qcut(df['OC'], q=4, labels=False)  # 4 bins

    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in strat_split.split(df, df['OC_bin']):
        strat_train_set = df.loc[train_idx]
        strat_test_set = df.loc[test_idx]

    # Drop the bin column
    strat_train_set.drop('OC_bin', axis=1, inplace=True)
    strat_test_set.drop('OC_bin', axis=1, inplace=True)

    # --------------------------------------------------
    # 3) DEFINE FEATURES AND TARGET
    # --------------------------------------------------
    X_train = strat_train_set.drop('OC', axis=1)
    y_train = strat_train_set['OC']
    X_test = strat_test_set.drop('OC', axis=1)
    y_test = strat_test_set['OC']

    # --------------------------------------------------
    # 4) LOG TRANSFORM THE TARGET
    #    We'll train on log(OC + 1) to handle skew
    # --------------------------------------------------
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # --------------------------------------------------
    # 5) WEIGHTED TRAINING FOR HIGH-OC BIN
    #    Re-bin the TRAIN set so we know which are high bin
    # --------------------------------------------------
    # (We re-create bins just for the training set.)
    # You can choose 3 if you want to up-weight the top 25% only
    train_bins = pd.qcut(y_train, q=4, labels=False)
    # Example: Give 3x weight to the highest bin (bin=3)
    sample_weights = train_bins.apply(lambda b: 3 if b == 3 else 1)

    # --------------------------------------------------
    # 6) HYPERPARAMETER TUNING with GridSearchCV
    #    We'll do a small grid for demonstration
    # --------------------------------------------------
    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb_model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    )

    grid_search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=3,  # 3-fold CV on the training set
        scoring='neg_root_mean_squared_error',
        verbose=1
    )

    # IMPORTANT: We pass sample_weight to .fit() via fit_params in GridSearchCV
    grid_search.fit(
        X_train, 
        y_train_log,
        **{'sample_weight': sample_weights}
    )

    print("\n=== Best Params from GridSearch ===")
    print(grid_search.best_params_)

    # Retrieve the best model
    best_model = grid_search.best_estimator_

    # --------------------------------------------------
    # 7) FINAL EVALUATION on Test Set
    # --------------------------------------------------
    # Predict in log space, then invert
    y_pred_log = best_model.predict(X_test)
    y_pred_inversed = np.expm1(y_pred_log)  # back to original scale

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_inversed))
    mae = mean_absolute_error(y_test, y_pred_inversed)
    r2 = r2_score(y_test, y_pred_inversed)
    print("\n=== Final Model Performance (Test Set) ===")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R²:   {r2:.2f}")

    # Mean Absolute Percentage Error
    residuals = y_test - y_pred_inversed
    mape = np.mean(np.abs(residuals / y_test)) * 100
    print(f"MAPE: {mape:.2f}%")

    # --------------------------------------------------
    # 8) ANALYZE ERRORS BY BIN in the TEST SET
    # --------------------------------------------------
    # Re-bin the TEST set to see how we do across bins
    strat_test_set = strat_test_set.copy()  # avoid SettingWithCopyWarning
    strat_test_set['Predicted'] = y_pred_inversed
    strat_test_set['AbsError'] = np.abs(strat_test_set['OC'] - strat_test_set['Predicted'])
    # Binning test set into 4 quartiles again
    strat_test_set['OC_bin'] = pd.qcut(strat_test_set['OC'], q=4, labels=False)

    bin_errors = strat_test_set.groupby('OC_bin')['AbsError'].mean()
    print("\nMean Absolute Error per OC Bin (Test Set):")
    print(bin_errors)

    # Visualize the absolute error distribution across bins
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='OC_bin', y='AbsError', data=strat_test_set)
    plt.title("Absolute Error by OC Bin (Weighted + Log + Tuned)")
    plt.xlabel("OC Bin (0 = lowest, 3 = highest)")
    plt.ylabel("Absolute Error (g/kg)")
    plt.show()

    # Plot predicted vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='red', alpha=0.6, label='Actual')
    plt.scatter(range(len(y_pred_inversed)), y_pred_inversed, color='blue', alpha=0.6, label='Predicted')
    plt.title("Predicted vs. Actual OC (Weighted + Log + Tuned)")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Organic Carbon (g/kg)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
