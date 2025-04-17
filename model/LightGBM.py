"""
improved_lgbm_stratified_optuna.py

This script demonstrates a pipeline for predicting soil Organic Carbon (OC)
with the following steps:
1. Data loading and preprocessing (using a processed CSV that has dropped rows with "< LOD").
2. Log-transforming the target (OC) to reduce skew.
3. Creating a stratified train-test split based on OC quartiles.
4. Hyperparameter tuning of a LightGBM regressor using Optuna.
5. Evaluating performance on the test set (with RMSE, MAE, R², and MAPE).
6. Visualizing predictions and error distributions.
7. Saving the tuned model for future use.

Author: [Your Name]
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

# -----------------------------
# 1. Load Data
# -----------------------------
df = pd.read_csv('LUCAS-SOIL-2018_processed.csv')  
# -----------------------------
# 2. Stratified Split Based on OC Quartiles
# -----------------------------
df['OC_bin'] = pd.qcut(df['OC'], q=4, labels=False)
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in strat_split.split(df, df['OC_bin']):
    strat_train_set = df.loc[train_idx].copy()
    strat_test_set = df.loc[test_idx].copy()
strat_train_set.drop('OC_bin', axis=1, inplace=True)
strat_test_set.drop('OC_bin', axis=1, inplace=True)

# -----------------------------
# 3. Define Features and Target
# -----------------------------
feature_cols = ['pH_H2O', 'EC', 'P', 'N', 'K', 'Elev']
X_train = strat_train_set[feature_cols]
y_train = strat_train_set['OC_log']
X_test = strat_test_set[feature_cols]
y_test = strat_test_set['OC_log']

# -----------------------------
# 4. Hyperparameter Tuning with Optuna
# -----------------------------
def objective(trial):
    # Define hyperparameters to tune
    param = {
        'num_leaves': trial.suggest_int('num_leaves', 31, 70),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'random_state': 42
    }
    
    model = lgb.LGBMRegressor(**param, verbose=-1)
    # Use 3-fold cross-validation on the training set with RMSE (we use negative RMSE to minimize)
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring=make_scorer(mean_squared_error, greater_is_better=False, squared=False))
    return -np.mean(cv_scores)

# Create an Optuna study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print("\n=== Best Parameters from Optuna ===")
print(study.best_params)

# Build the best model using the tuned parameters
best_params = study.best_params
best_params['random_state'] = 42
best_model = lgb.LGBMRegressor(**best_params)
best_model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate the Model on Test Data
# -----------------------------
y_pred_log = best_model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # Invert log transform: exp(x) - 1
y_true = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(f"\nTest RMSE: {rmse:.3f}")
print(f"Test MAE:  {mae:.3f}")
print(f"Test R²:   {r2:.3f}")

epsilon = 1e-3
mape = np.mean(100.0 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon))
print(f"MAPE: {mape:.2f}%")

# -----------------------------
# 6. Visualization: Sorted Plot (Actual vs. Predicted)
# -----------------------------
y_true_array = np.array(y_true)
y_pred_array = np.array(y_pred)
sorted_idx = np.argsort(y_true_array)

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(range(len(y_true_array)), y_true_array[sorted_idx], 'o-', alpha=0.7, label='Actual (Sorted)')
plt.plot(range(len(y_pred_array)), y_pred_array[sorted_idx], 'o-', alpha=0.7, label='Predicted (Sorted)')
plt.title("OC: Actual vs. Predicted (Sorted by Actual)")
plt.xlabel("Sorted Sample Index")
plt.ylabel("Organic Carbon (g/kg)")
plt.legend()
plt.show()

# -----------------------------
# 7. Visualization: Histogram of Absolute Percentage Errors (APE)
# -----------------------------
ape = 100.0 * np.abs(y_pred_array - y_true_array) / np.maximum(np.abs(y_true_array), epsilon)
plt.figure(figsize=(8, 5))
plt.hist(ape, bins=10, alpha=0.7)
plt.title("Distribution of Absolute Percentage Errors")
plt.xlabel("Absolute Percentage Error (%)")
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# 8. Detailed Analysis: Top 20 Rows with Highest Percentage Error
# -----------------------------
test_df = X_test.copy()
test_df['OC_actual'] = y_true
test_df['OC_pred'] = y_pred
test_df['APE'] = 100.0 * np.abs(test_df['OC_pred'] - test_df['OC_actual']) / (np.abs(test_df['OC_actual']) + epsilon)
test_df_sorted = test_df.sort_values(by='APE', ascending=False)
print("\nTop 20 Rows with Highest Percentage Error:")
print(test_df_sorted.head(20))
highest_error_indices = test_df_sorted.head(20).index
print("Indices with highest percent error:", highest_error_indices.tolist())

# -----------------------------
# 9. Visualization: Residual Distribution
# -----------------------------
residuals = y_pred_array - y_true_array
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, alpha=0.7)
plt.xlabel("Residual (Predicted - Actual)")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals (Original Scale)")
plt.show()

# -----------------------------
# 10. Save the Tuned Model for Future Use
# -----------------------------
joblib.dump(best_model, 'lgbm_model1.pkl')
print("Model saved as 'lgbm_model.pkl'")
