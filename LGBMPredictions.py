import joblib
import pandas as pd
import numpy as np

# Load the saved LightGBM model
model = joblib.load('lgbm_model.pkl')
print("Model loaded successfully.")

# Define dummy values for the features (pH_H2O, EC, P, N, K, Elev)
# Adjust these values as needed to represent a typical sample
dummy_values = {
    'pH_H2O': 7.0,   # typical pH value
    'EC': 20.0,      # example electrical conductivity (mS/m)
    'P': 15.0,       # phosphorus (mg/kg)
    'N': 1.0,        # nitrogen (g/kg)
    'K': 50.0,       # potassium (mg/kg)
    'Elev': 100.0    # elevation (m)
}

# Convert the dummy values into a one-row DataFrame
dummy_df = pd.DataFrame([dummy_values])
print("Dummy input:")
print(dummy_df)

# Use the loaded model to predict the log-transformed TOC
y_pred_log = model.predict(dummy_df)

# Invert the log transform (if your model was trained on log(OC + 1))
y_pred = np.expm1(y_pred_log)

print(f"\nPredicted Organic Carbon (OC): {y_pred[0]:.2f} g/kg")
