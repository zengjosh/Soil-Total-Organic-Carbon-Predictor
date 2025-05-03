#!/usr/bin/env python3
import os
import sys
import json
import joblib
import numpy as np
import random
import time

# -------------------------------------------------------
# 1) Determine the path to this script�s directory,
#    then build the absolute path to your .pkl file.
# -------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = 'lgbm_model.pkl'  # <-- Update if your file is named differently
MODEL_PATH = os.path.join(THIS_DIR, MODEL_NAME)

# -------------------------------------------------------
# 2) Load the model once
# -------------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    sys.stderr.write(f"ERROR loading model at {MODEL_PATH}: {e}\n")
    sys.exit(1)

# -------------------------------------------------------
# 3) Generate random sensor data
# -------------------------------------------------------
sensor_data = {
    "pH_H2O": round(random.uniform(5.5, 8.5), 2),  # realistic pH range
    "EC":     round(random.uniform(10, 50), 2),    # electrical conductivity
    "P":      round(random.uniform(5, 30), 2),     # phosphorus
    "N":      round(random.uniform(0.5, 5), 2),    # nitrogen
    "K":      round(random.uniform(20, 100), 2),   # potassium
    "Elev":   round(random.uniform(0, 500), 2)     # elevation
}

# -------------------------------------------------------
# 4) Prepare feature matrix in the same column order you trained on
# -------------------------------------------------------
features = ["pH_H2O", "EC", "P", "N", "K", "Elev"]
X = np.array([[sensor_data[k] for k in features]])

# -------------------------------------------------------
# 5) Predict (in log space), invert the log transform
# -------------------------------------------------------
y_log = model.predict(X)
prediction = float(np.expm1(y_log))

# -------------------------------------------------------
# 6) Merge and output JSON
# -------------------------------------------------------
sensor_data["carbonContent"] = prediction
print(json.dumps(sensor_data))
