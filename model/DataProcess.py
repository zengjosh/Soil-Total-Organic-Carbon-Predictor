import pandas as pd
import numpy as np

# For additional transformations
from sklearn.preprocessing import PowerTransformer, StandardScaler

# -----------------------------
# 1. Load the raw dataset
# -----------------------------
df = pd.read_csv('LUCAS-SOIL-2018.csv')

# Keep only the relevant columns
cols = ['pH_H2O', 'EC', 'P', 'N', 'K', 'Elev', 'OC']
df = df[cols].copy()

# -----------------------------
# 2. Replace "< LOD" values for P, N, K with NaN so they can be handled
# -----------------------------
lod_dict = {'P': 10, 'N': 0.2, 'K': 10}
for col, lod in lod_dict.items():
    # Replace any occurrences of "< LOD" with NaN
    df[col] = df[col].replace(r'\s*<\s*LOD\s*', None, regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Ensure the other columns are numeric
for col in ['pH_H2O', 'EC', 'Elev', 'OC']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# -----------------------------
# 3. Drop rows with missing values
# -----------------------------
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# -----------------------------
# 4. Outlier Capping (optional)
# -----------------------------
# Cap extreme values at the 99th percentile to mitigate undue influence of outliers
def cap_outliers(series, percentile=99):
    cap = np.percentile(series, percentile)
    return np.where(series > cap, cap, series)

# Uncomment the following lines to cap outliers for predictors and/or target
# For example, to cap 'OC' and 'EC' (you may repeat for others):
df['OC'] = cap_outliers(df['OC'], 99)

# -----------------------------
# 5. Log-Transform the Target (OC)
# -----------------------------
df['OC_log'] = np.log1p(df['OC'])

# -----------------------------
# 6. Alternative Transformation for Predictors (Yeo-Johnson)
# -----------------------------

predictors = ['pH_H2O', 'EC', 'P', 'N', 'K', 'Elev']
pt = PowerTransformer(method='yeo-johnson')
df[predictors] = pt.fit_transform(df[predictors])
#
# Note: This transformation can alter the scale and distribution of your features.
# Compare model performance with and without this step.

# -----------------------------
# 7. (Optional) Standard Scaling of Predictors
# -----------------------------
# If you want to standardize your features, you could also apply StandardScaler.
# This is less crucial for tree-based models like LightGBM but might be useful for other models.
#
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[predictors] = scaler.fit_transform(df[predictors])

# -----------------------------
# 8. Save the Processed Data
# -----------------------------
print("Descriptive statistics after preprocessing:")
print(df.describe())

df.to_csv('LUCAS-SOIL-2018_processed.csv', index=False)
print("Preprocessed data saved as 'LUCAS-SOIL-2018_processed.csv'")
