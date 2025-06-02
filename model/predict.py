#!/usr/bin/env python3
import os
import sys
import json
import joblib
import numpy as np
import time
import serial
import pandas as pd
import subprocess
subprocess.run(["stty", "-F", "/dev/ttyUSB0", "-hupcl"])

# --------------------------------------------
# 1) Load the LightGBM model
# --------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(THIS_DIR, 'lgbm_model.pkl')

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    sys.stderr.write(f"ERROR loading model: {e}\n")
    sys.exit(1)

# --------------------------------------------
# 2) Open serial port to Arduino
# --------------------------------------------
try:
    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=2)
    time.sleep(2)
except Exception as e:
    sys.stderr.write(f"ERROR opening serial port: {e}\n")
    sys.exit(1)

# --------------------------------------------
# 3) Parse CSV input 
# --------------------------------------------
def parse_csv_line(line):
    try:
        values = list(map(float, line.strip().split(',')))
        if len(values) != 6:
            raise ValueError(f"Expected 6 values, got {len(values)}")
        raw_keys = ["pH", "EC", "N", "P", "K", "Elev"]
        raw_data = dict(zip(raw_keys, values))

        # Remap for model input
        sensor_data = {
            "pH_H2O": raw_data["pH"],  # for model
            "EC":     raw_data["EC"],
            "P":      raw_data["P"],
            "N":      raw_data["N"],
            "K":      raw_data["K"],
            "Elev":   raw_data["Elev"]
        }
        return sensor_data
    except Exception as e:
        sys.stderr.write(f"ERROR parsing line: {line} -- {e}\n")
        return None

sensor_data = None
for _ in range(10):
    try:
        line = ser.readline().decode('utf-8').strip()
        if line:
            sensor_data = parse_csv_line(line)
            if sensor_data:
                break
    except Exception as e:
        sys.stderr.write(f"Serial read error: {e}\n")

ser.close()

if not sensor_data:
    sys.stderr.write("Failed to read valid sensor data from Arduino.\n")
    sys.exit(1)

# --------------------------------------------
# 4) Predict OC and output
# --------------------------------------------
feature_keys = ["pH_H2O", "EC", "P", "N", "K", "Elev"]
try:
    df = pd.DataFrame([sensor_data], columns=feature_keys)
    y_log = model.predict(df)
    prediction = float(np.expm1(y_log[0]))
    sensor_data["carbonContent"] = prediction
except Exception as e:
    sys.stderr.write(f"Prediction failed: {e}\n")
    sys.exit(1)

print(json.dumps(sensor_data))
