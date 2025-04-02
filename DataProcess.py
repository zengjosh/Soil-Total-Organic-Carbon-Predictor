import pandas as pd
import numpy as np

# Load the raw dataset
df = pd.read_csv('LUCAS-SOIL-2018.csv')

# Keep only the relevant columns
cols = ['pH_H2O', 'EC', 'P', 'N', 'K', 'Elev', 'OC']
df = df[cols].copy()

# Replace "< LOD" values for P, N, K with half the LOD
lod_dict = {'P': 10, 'N': 0.2, 'K': 10}
for col, lod in lod_dict.items():
    df[col] = df[col].replace(r'\s*<\s*LOD\s*', None, regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Ensure the other columns are numeric
for col in ['pH_H2O','EC','Elev','OC']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)

# Log-transform the OC column to handle skewed distribution
df['OC_log'] = np.log1p(df['OC'])

print(df.count())  # Check the count of non-null values in each column

# Save the processed data to a new CSV file
df.to_csv('LUCAS-SOIL-2018_processed.csv', index=False)
print("Preprocessed data saved as 'LUCAS-SOIL-2018_processed.csv'")
