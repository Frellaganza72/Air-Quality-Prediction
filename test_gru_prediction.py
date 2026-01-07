#!/usr/bin/env python3
"""Test GRU prediction to debug high values"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Setup paths
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from utils.prediction import PredictionEngine
from utils.ispu_classifier import ISPUClassifier
import joblib
import os

# Paths
BASE_DIR = Path(__file__).parent / "backend"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
TRAINING_DIR = BASE_DIR / "Training"

# Load GRU model and scaler
print("Loading GRU model and scalers...")
gru_model_path = MODEL_DIR / "GRU" / "model_v9.keras"
gru_scaler_path = MODEL_DIR / "GRU" / "scaler_X.pkl"
gru_y_scaler_path = MODEL_DIR / "GRU" / "scalers_y.pkl"

try:
    import tensorflow as tf
    gru_model = tf.keras.models.load_model(str(gru_model_path))
    print(f"✅ GRU model loaded: {gru_model_path}")
except Exception as e:
    print(f"❌ GRU model load error: {e}")
    gru_model = None

try:
    gru_scaler = joblib.load(str(gru_scaler_path))
    print(f"✅ GRU scaler loaded: {gru_scaler_path}")
except Exception as e:
    print(f"❌ GRU scaler load error: {e}")
    gru_scaler = None

try:
    gru_y_scaler = joblib.load(str(gru_y_scaler_path))
    print(f"✅ GRU y_scaler loaded: {gru_y_scaler_path}")
    if isinstance(gru_y_scaler, dict):
        print(f"   Scaler keys: {list(gru_y_scaler.keys())}")
except Exception as e:
    print(f"❌ GRU y_scaler load error: {e}")
    gru_y_scaler = None

# Load preprocessed CSV
preprocessed_csv = DATA_DIR / "dataset_preprocessed" / "dataset_preprocessed.csv"

# Initialize prediction engine
print("\nInitializing Prediction Engine...")
try:
    pred_engine = PredictionEngine(
        dt_model=None,
        dt_scaler=None,
        cnn_model=None,
        cnn_scaler=None,
        cnn_y_scaler=None,
        gru_model=gru_model,
        gru_scaler=gru_scaler,
        y_scaler=gru_y_scaler,
        preprocessed_csv=str(preprocessed_csv) if preprocessed_csv.exists() else None
    )
    print("✅ Prediction engine initialized")
except Exception as e:
    print(f"❌ Prediction engine init error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test data
test_data = {
    'pm2_5 (μg/m³)_mean': 20.0,
    'ozone (μg/m³)_mean': 55.0,
    'carbon_monoxide (μg/m³)_mean': 600.0,
    'temperature_2m (°C)_mean': 27.0,
    'relative_humidity_2m (%)_mean': 75.0,
    'wind_speed_10m (km/h)_mean': 10.0,
    'surface_pressure (hPa)_mean': 1013.0,
    'hari': 15,
    'bulan': 1,
}

print(f"\nTest Data: {test_data}")
print("\nCalling predict_gru...")
try:
    result = pred_engine.predict_gru(test_data)
    print(f"\n✅ GRU Prediction Result:")
    print(f"   PM2.5: {result['pm25']:.2f}")
    print(f"   O3:    {result['o3']:.2f}")
    print(f"   CO:    {result['co']:.2f}")
except Exception as e:
    print(f"❌ Prediction error: {e}")
    import traceback
    traceback.print_exc()
