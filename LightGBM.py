import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

# ---------------------------------------------------------------------
# 1. Read and Data
# ---------------------------------------------------------------------
df = pd.read_csv('LUCAS-SOIL-2018_processed.csv')

# ---------------------------------------------------------------------
# 2. Log-Transform the Target (helps with skewed OC)
# ---------------------------------------------------------------------
df['OC_log'] = np.log1p(df['OC'])

# ---------------------------------------------------------------------
# 3. Create Features & Target
# ---------------------------------------------------------------------
feature_cols = ['pH_H2O', 'EC', 'P', 'N', 'K', 'Elev']
X = df[feature_cols]
y = df['OC_log']

# ---------------------------------------------------------------------
# 4. Train-Test Split (Basic 80/20 here)
#    - If you want to ensure representation of high OC, 
#      you could do quantile-based stratification as shown previously.
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------------------------
# 5. Train LightGBM Regressor
# ---------------------------------------------------------------------
model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------
# 6. Predictions on Test Set (Inverse Log to Original Scale)
# ---------------------------------------------------------------------
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

# ---------------------------------------------------------------------
# 7. Evaluate Performance (RMSE, MAE, R^2)
# ---------------------------------------------------------------------
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)

print(f"Test RMSE: {rmse:.3f}")
print(f"Test MAE:  {mae:.3f}")
print(f"Test R^2:  {r2:.3f}")

# ---------------------------------------------------------------------
# 8. Visualization: Predicted vs. Actual (Scatter + 45° Reference Line)
# ---------------------------------------------------------------------
# plt.figure()
# plt.scatter(y_true, y_pred, alpha=0.5)
# # Perfect-prediction reference line
# min_val = min(min(y_true), min(y_pred))
# max_val = max(max(y_true), max(y_pred))
# plt.plot([min_val, max_val], [min_val, max_val], 'r--')
# plt.xlabel("Actual OC")
# plt.ylabel("Predicted OC")
# plt.title("Predicted vs. Actual OC (Original Scale)")
# plt.show()

# ---------------------------------------------------------------------
# 9. Visualization: Sorted Plot (Actual vs. Predicted)
# ---------------------------------------------------------------------
# Convert to NumPy arrays to avoid indexing issues.
y_true_array = np.array(y_true)
y_pred_array = np.array(y_pred)

# Sort indices by ascending actual value
sorted_idx = np.argsort(y_true_array)

# Turn up DPI in final figure
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(range(len(y_true_array)), y_true_array[sorted_idx], 'o-', alpha=0.7, label='Actual (Sorted)')
plt.plot(range(len(y_true_array)), y_pred_array[sorted_idx], 'o-', alpha=0.7, label='Predicted (Sorted)')
plt.title("OC: Actual vs. Predicted (Sorted by Actual)")
plt.xlabel("Sorted Sample Index")
plt.ylabel("OC (Original Scale)")
plt.legend()
plt.show()


ape = 100.0 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1e-6)

# 2. Compute Mean Absolute Percentage Error (MAPE)
mape = np.mean(ape)
print(f"MAPE: {mape:.2f}%")

# 3. Plot Histogram of APE (percentage errors)
plt.figure(figsize=(8, 5))
plt.hist(ape, bins=10, alpha=0.7)
plt.title("Distribution of Absolute Percentage Errors")
plt.xlabel("Absolute Percentage Error (%)")
plt.ylabel("Frequency")
plt.show()

# 1. After your train_test_split, store the test index and test features in a DataFrame
test_df = X_test.copy()
test_df['OC_actual'] = y_true   # y_true is np.expm1(y_test) from your model code
test_df['OC_pred']   = y_pred   # y_pred is np.expm1(y_pred_log)
 
# 2. Calculate Absolute Percentage Error (APE)
#    to avoid division by zero, use a small epsilon if needed
epsilon = 1e-6
test_df['APE'] = 100.0 * abs(test_df['OC_pred'] - test_df['OC_actual']) / (abs(test_df['OC_actual']) + epsilon)

# 3. Sort by APE descending (highest error first)
test_df_sorted = test_df.sort_values(by='APE', ascending=False)

# 4. Display top 10 rows with largest percentage errors
print(test_df_sorted.head(20))

# If you want just the row indices:
highest_error_indices = test_df_sorted.head(20).index
print("Indices with highest percent error:", highest_error_indices)


# ---------------------------------------------------------------------
# 10. Visualization: Residual Distribution (Predicted - Actual)
# ---------------------------------------------------------------------
residuals = y_pred_array - y_true_array
plt.figure()
plt.hist(residuals, bins=30, alpha=0.7)
plt.xlabel("Residual (Predicted - Actual)")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals (Original Scale)")
plt.show()
