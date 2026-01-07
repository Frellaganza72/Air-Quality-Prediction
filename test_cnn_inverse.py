#!/usr/bin/env python
"""Test CNN prediction with inverse transform."""
import sys
sys.path.insert(0, '/Users/user/Desktop/SKRIPSI/HASIL/Air Quality/backend')

from utils.prediction import quick_load_engine
import json
from datetime import datetime, date

# Load the engine
print("Loading prediction engine...")
engine = quick_load_engine('/Users/user/Desktop/SKRIPSI/HASIL/Air Quality/backend/models')

# Check if cnn_y_scaler is loaded
print(f"CNN y_scaler loaded: {engine.cnn_y_scaler is not None}")
if engine.cnn_y_scaler is not None:
    print(f"CNN y_scaler type: {type(engine.cnn_y_scaler)}")
    print(f"CNN y_scaler: {engine.cnn_y_scaler}")

# Simulate today's data (use last known data)
test_data = {
    'temperature (°C)_mean': 26.5,
    'humidity (%RH)_mean': 72.0,
    'pressure (hPa)_mean': 1012.5,
    'wind_speed (m/s)_mean': 2.1,
    'wind_direction (°)_mean': 180.0,
    'pm2_5 (μg/m³)_mean': 25.0,
    'ozone (μg/m³)_mean': 60.0,
    'carbon_monoxide (μg/m³)_mean': 800.0,
    'no2 (μg/m³)_mean': 40.0,
    'so2 (μg/m³)_mean': 15.0,
    'hari': float(date.today().day),
    'bulan': float(date.today().month),
}

print("\n" + "="*60)
print("Testing CNN Prediction")
print("="*60)
print(f"Input data: {test_data}")

cnn_pred = engine.predict_cnn(test_data)
print(f"\nCNN Prediction: {cnn_pred}")
print(f"  PM2.5: {cnn_pred['pm25']:.2f} μg/m³")
print(f"  O3: {cnn_pred['o3']:.2f} μg/m³")
print(f"  CO: {cnn_pred['co']:.2f} μg/m³")

# Also test Decision Tree and GRU for comparison
print("\n" + "="*60)
dt_pred = engine.predict_dt(test_data)
print(f"Decision Tree Prediction: {dt_pred}")
print(f"  PM2.5: {dt_pred['pm25']:.2f} μg/m³")

gru_pred = engine.predict_gru(test_data)
print(f"\nGRU Prediction: {gru_pred}")
print(f"  PM2.5: {gru_pred['pm25']:.2f} μg/m³")
