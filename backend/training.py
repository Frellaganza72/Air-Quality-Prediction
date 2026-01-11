#!/usr/bin/env python3
"""
Training script for Air Quality Prediction Models
Extracted from Training_Rehan.ipynb

This script includes:
1. Data preprocessing
2. Decision Tree training
3. CNN multi-head model training
4. GRU multi-output model training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from scipy import stats
from pathlib import Path
import joblib
import os
import warnings
import json
import math
import logging

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available, deep learning models will be skipped")
    TENSORFLOW_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
PREPROCESSED_DIR = DATA_DIR / "dataset_preprocessed"
RESULTS_DIR = BASE_DIR / "Training"

# Model output directories (untuk training artifacts)
DT_DIR = RESULTS_DIR / "Decision Tree"
GRU_DIR = RESULTS_DIR / "GRU"
CNN_DIR = RESULTS_DIR / "CNN"
SHARED_DIR = RESULTS_DIR / "shared_data"

# Models directory (untuk model final yang digunakan aplikasi)
MODELS_DIR = BASE_DIR / "models"
MODELS_DT_DIR = MODELS_DIR / "Decision Tree"
MODELS_CNN_DIR = MODELS_DIR / "CNN"
MODELS_GRU_DIR = MODELS_DIR / "GRU"

# Data file paths
CUACA_PATH = RAW_DATA_DIR / "Data Cuaca 2023-2025.xlsx"
POLUT_PATH = RAW_DATA_DIR / "Data Polutan.xlsx"

# Create directories
for folder in [DT_DIR, GRU_DIR, CNN_DIR, SHARED_DIR, PREPROCESSED_DIR, 
               MODELS_DT_DIR, MODELS_CNN_DIR, MODELS_GRU_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
if TENSORFLOW_AVAILABLE:
    tf.random.set_seed(SEED)

def prepare_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Standarisasi kolom tanggal/waktu"""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()

    if df.index.name and isinstance(df.index.name, str):
        idx_name = df.index.name.strip().lower()
        if any(k in idx_name for k in ["tanggal", "date", "time", "waktu", "datetime"]):
            df = df.reset_index()

    date_col = None
    for col in df.columns:
        if any(k in col for k in ["tanggal", "date", "time", "waktu", "datetime"]):
            date_col = col
            break

    if date_col is None:
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[~df[date_col].isna()].copy()
    df = df.rename(columns={date_col: "tanggal_waktu"})

    return df.sort_values("tanggal_waktu")

def find_pollutant_columns(columns):
    """Deteksi kolom polutan dengan perbaikan untuk CO"""
    cols_lower = [str(c).lower() for c in columns]
    targets = []

    for c in cols_lower:
        # PM2.5
        if ("pm2.5" in c) or ("pm2_5" in c) or ("pm2" in c and "temp" not in c):
            targets.append(c)

        # Ozone (O3)
        if ("o3" in c) or ("ozon" in c) or ("ozone" in c):
            targets.append(c)

        # Carbon Monoxide (CO)
        # keep heuristics but avoid cloud/cover false positives
        cond = (
            ("carbon_monoxide" in c) or
            ("carbon monoxide" in c) or
            (c == "co") or
            c.endswith("_co") or
            (" co " in c) or
            ((c.startswith("co ") or c.endswith(" co")) and ("cloud" not in c and "cover" not in c))
        )
        if cond:
            targets.append(c)

    final = []
    for t in targets:
        for orig in columns:
            if str(orig).lower() == t and orig not in final:
                final.append(orig)

    return final

def season_from_month(m):
    """Konversi bulan ke musim"""
    if m in [12, 1, 2]: return 1
    if m in [3, 4, 5]: return 2
    if m in [6, 7, 8]: return 3
    return 4


# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_data():
    """Step 1-12: Complete data preprocessing pipeline"""
    print("\n" + "="*80)
    print("PREPROCESSING -> SAVE AS PICKLE (joblib)")
    print("Output folder:", SHARED_DIR.resolve())
    print("="*80)

    # 1. Load data
    print("\n[1/12] Loading data...")
    df_cuaca = prepare_date_column(pd.read_excel(CUACA_PATH))
    df_polut_raw = prepare_date_column(pd.read_excel(POLUT_PATH))
    print(f"✅ Data cuaca: {len(df_cuaca)} records")
    print(f"✅ Data polutan: {len(df_polut_raw)} records")

    # 2. Aggregate pollutant to daily
    print("\n[2/12] Agregasi polutan per hari...")
    df_polut_raw["tanggal"] = pd.to_datetime(df_polut_raw["tanggal_waktu"].dt.date)
    df_polut = df_polut_raw.drop(columns=["tanggal_waktu"])

    df_polut_agg = df_polut.groupby("tanggal").agg(["mean", "max", "median"]).reset_index()
    df_polut_agg.columns = ["_".join([c for c in col if c]).strip("_") for col in df_polut_agg.columns.values]
    df_polut_agg = df_polut_agg.rename(columns={"tanggal_": "tanggal"})
    print(f"✅ Agregasi selesai: {len(df_polut_agg)} hari")

    # 3. Convert weather to daily
    print("\n[3/12] Konversi cuaca ke harian...")
    df_cuaca["tanggal"] = pd.to_datetime(df_cuaca["tanggal_waktu"].dt.date)
    df_cuaca = df_cuaca.drop(columns=["tanggal_waktu"])

    # 4. Merge
    print("\n[4/12] Merge polutan + cuaca...")
    df = pd.merge(df_polut_agg, df_cuaca, on="tanggal", how="inner")
    if df.empty:
        df = pd.merge(df_polut_agg, df_cuaca, on="tanggal", how="outer").sort_values("tanggal")
    print(f"✅ Total data: {len(df)} hari ({df['tanggal'].min()} → {df['tanggal'].max()})")

    # 5. Drop columns with >30% missing
    print("\n[5/12] Drop kolom >30% missing...")
    missing_ratio = df.isna().mean()
    drop_cols = missing_ratio[missing_ratio > 0.30].index.tolist()
    if drop_cols:
        print("⚠️ Kolom dihapus:", drop_cols)
        df.drop(columns=drop_cols, inplace=True)
    else:
        print("✅ Tidak ada kolom dihapus")

    # 6. Interpolate & impute
    print("\n[6/12] Interpolasi & imputasi...")
    df = df.sort_values("tanggal").reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    non_num_cols = df.select_dtypes(exclude=[np.number, "datetime64[ns]"]).columns.tolist()
    if non_num_cols:
        df[non_num_cols] = df[non_num_cols].ffill().bfill()
        for col in non_num_cols:
            if df[col].isna().any():
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")

    print("✅ Missing after impute:", int(df.isna().sum().sum()))

    # 7. Outliers detection
    print("\n[7/12] Detect outliers...")
    df_outliers = pd.DataFrame()
    if len(df) >= 30 and numeric_cols:
        z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy="omit"))
        outlier_mask = np.any(z_scores > 3, axis=1)

        df_outliers = df[outlier_mask].copy()
        df_clean = df[~outlier_mask].copy()

        print(f"⚠️ Outliers found: {len(df_outliers)} rows")
        print(f"✅ Cleaned rows: {len(df_clean)}")
        df = df_clean.reset_index(drop=True)
    else:
        print("✅ Not enough rows or no numeric columns to detect outliers")

    # Save outliers (both CSV and pickle)
    df_outliers.to_csv(SHARED_DIR / "outliers_anomaly_candidates.csv", index=False)
    joblib.dump(df_outliers, SHARED_DIR / "outliers_anomaly_candidates.pkl")

    # 8. Feature engineering
    print("\n[8/12] Feature engineering...")
    df["hari"] = df["tanggal"].dt.dayofweek
    df["bulan"] = df["tanggal"].dt.month
    df["musim"] = df["bulan"].apply(season_from_month)
    df["is_weekend"] = df["hari"].isin([5, 6]).astype(int)

    if {"temp", "humidity"}.issubset(df.columns):
        df["temp_x_humidity"] = df["temp"] * df["humidity"]

    if {"windspeed", "humidity"}.issubset(df.columns):
        df["wind_x_humidity"] = df["windspeed"] * df["humidity"]

    print("✅ fitur temporal dibuat")

    # 9. Pollutant identification
    print("\n[9/12] Identifikasi kolom polutan...")
    pollutants = find_pollutant_columns(df.columns)
    print(f"✅ pollutant cols: {pollutants}")
    co_detected = any("carbon" in p.lower() or "monoxide" in p.lower() for p in pollutants)
    print(f"{('✅' if co_detected else '⚠️ ')} CO detected: {co_detected}")

    # 10. Scaling
    print("\n[10/12] Scaling numeric features (MinMax) — skip pollutant cols...")
    scale_features = [c for c in numeric_cols if c not in pollutants and df[c].nunique() > 1]
    scaler = MinMaxScaler()
    if scale_features:
        df[scale_features] = scaler.fit_transform(df[scale_features])
    print(f"✅ scaled features count: {len(scale_features)}")

    # Save scaler
    joblib.dump({
        "scaler": scaler,
        "scaled_features": scale_features,
        "pollutant_cols": pollutants
    }, SHARED_DIR / "minmax_scaler.pkl")

    # Also save as CSV for compatibility
    df.to_csv(PREPROCESSED_DIR / "dataset_preprocessed.csv", index=False)
    joblib.dump(scaler, PREPROCESSED_DIR / "minmax_scaler.pkl")

    # 11. Split dataset
    print("\n[11/12] Split dataset (70/15/15)...")
    n = len(df)
    test_n = int(np.floor(0.15 * n))
    val_n = int(np.floor(0.15 * n))
    train_end = n - (test_n + val_n)
    val_end = n - test_n

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"✅ Train/Val/Test = {len(train_df)}/{len(val_df)}/{len(test_df)} rows")

    # 12. Save everything as pickles
    print("\n[12/12] Menyimpan data (CSV + pickles)...")

    # Full preprocessed dataset - save as CSV
    df.to_csv(SHARED_DIR / "dataset_preprocessed.csv", index=False)

    # Splits - save as CSV
    train_df.to_csv(SHARED_DIR / "train.csv", index=False)
    val_df.to_csv(SHARED_DIR / "val.csv", index=False)
    test_df.to_csv(SHARED_DIR / "test.csv", index=False)

    # Outliers - save as both CSV and pickle
    df_outliers.to_csv(SHARED_DIR / "outliers_anomaly_candidates.csv", index=False)

    # Metadata
    metadata = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'date_range': (df['tanggal'].min().isoformat(), df['tanggal'].max().isoformat()) if len(df) else (None, None),
        'pollutant_cols': pollutants,
        'feature_cols': df.columns.tolist(),
        'target_col': next((p for p in pollutants if "pm2" in str(p).lower()), None),
        'split_info': {
            'train': len(train_df),
            'val': len(val_df),
            'test': len(test_df)
        },
        'artifacts': {
            'dataset_preprocessed': str(SHARED_DIR / "dataset_preprocessed.csv"),
            'train': str(SHARED_DIR / "train.csv"),
            'val': str(SHARED_DIR / "val.csv"),
            'test': str(SHARED_DIR / "test.csv"),
            'outliers': str(SHARED_DIR / "outliers_anomaly_candidates.csv")
        }
    }
    joblib.dump(metadata, SHARED_DIR / "preprocessing_metadata.pkl")

    print("\nSaved artifacts (CSV):")
    for path in [
        SHARED_DIR / "dataset_preprocessed.csv",
        SHARED_DIR / "train.csv",
        SHARED_DIR / "val.csv",
        SHARED_DIR / "test.csv",
        SHARED_DIR / "outliers_anomaly_candidates.csv",
        SHARED_DIR / "preprocessing_metadata.pkl"
    ]:
        print(" -", path)

    print("\n✅ Preprocessing complete!")
    return train_df, val_df, test_df, pollutants

# ============================================================================
# DECISION TREE TRAINING
# ============================================================================

def train_decision_tree():
    """Train Multi-Output Decision Tree model"""
    print("\n" + "="*80)
    print("DECISION TREE MODEL TRAINING (Multi-Output)")
    print("="*80)

    # 1. Load data
    print("\n[1/9] Loading data...")
    try:
        # Load from CSV
        train_df = pd.read_csv(SHARED_DIR / "train.csv")
        val_df = pd.read_csv(SHARED_DIR / "val.csv")
        test_df = pd.read_csv(SHARED_DIR / "test.csv")
        train_df['tanggal'] = pd.to_datetime(train_df['tanggal'], errors='coerce')
        val_df['tanggal'] = pd.to_datetime(val_df['tanggal'], errors='coerce')
        test_df['tanggal'] = pd.to_datetime(test_df['tanggal'], errors='coerce')
        print("✅ Loaded from CSV")
    except FileNotFoundError as e:
        print(f"❌ CSV files not found: {e}")
        raise
    print(f"✅ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2. Identify targets
    # Helper from GRU section (reused here or we can define it globally)
    def find_target_column(df, keywords):
        for kw in keywords:
            matches = [c for c in df.columns if kw in c.lower() and 'mean' in c.lower()]
            if matches: return matches[0]
        return None

    t_pm25 = find_target_column(train_df, ['pm2_5', 'pm2.5', 'pm2'])
    t_o3   = find_target_column(train_df, ['ozone', 'o3'])
    t_co   = find_target_column(train_df, ['carbon_monoxide', 'carbon monoxide', 'co'])
    
    target_cols = [t for t in [t_pm25, t_o3, t_co] if t is not None]
    if not target_cols:
        target_cols = ["pm2_5 (μg/m³)_mean"] # Fallback
    
    print(f"✅ Targets: {target_cols}")

    # 3. Feature engineering
    print("\n[2/9] Feature engineering...")

    def engineer_features(df, target_cols):
        """Create advanced features for multiple targets"""
        df = df.copy()

        # Polynomial features untuk semua polutan
        for target in target_cols:
            base_name = target.split('_mean')[0]
            median_col = f"{base_name}_median"
            max_col = f"{base_name}_max"

            if median_col in df.columns and max_col in df.columns:
                df[f'{base_name}_mean_median_ratio'] = df[target] / (df[median_col] + 1e-8)
                df[f'{base_name}_mean_max_ratio'] = df[target] / (df[max_col] + 1e-8)
                df[f'{base_name}_range'] = df[max_col] - df[median_col]

        # Interaction features
        if 'temp' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temp'] * df['humidity']
            df['heat_index'] = df['temp'] + 0.5 * df['humidity']

        if 'windspeed' in df.columns and 'humidity' in df.columns:
            df['wind_humidity_interaction'] = df['windspeed'] * df['humidity']

        # Pollutant interactions (if we have O3 and CO)
        if len(target_cols) >= 3:
             # Asumsi urutan [pm2.5, o3, co] atau kita cari by name
             pass # Logic spesifik bisa ditambahkan jika nama kolom pasti

        # Weather severity index
        if 'temp' in df.columns and 'humidity' in df.columns and 'windspeed' in df.columns:
            df['weather_severity'] = (df['temp'] / 100) + (df['humidity'] / 100) + (df['windspeed'] / 10)

        # Temporal features
        if 'hari' in df.columns and 'bulan' in df.columns:
            df['hari_bulan_interaction'] = df['hari'] * df['bulan']

        return df

    train_df = engineer_features(train_df, target_cols)
    val_df = engineer_features(val_df, target_cols)
    test_df = engineer_features(test_df, target_cols)
    print(f"✅ Feature engineering completed")

    # 4. Prepare features
    print("\n[3/9] Preparing features...")
    exclude_cols = ['tanggal'] + target_cols
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    X_train = train_df[feature_cols]
    y_train = train_df[target_cols]
    X_val = val_df[feature_cols]
    y_val = val_df[target_cols]
    X_test = test_df[feature_cols]
    y_test = test_df[target_cols]
    print(f"✅ Features: {len(feature_cols)}")

    # 5. Feature scaling
    print("\n[4/9] Scaling features...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Keep as DataFrame for convenience
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)

    joblib.dump(scaler, DT_DIR / "scaler.pkl")
    print("✅ Features scaled")

    # 6. Hyperparameter tuning
    print("\n[5/9] Hyperparameter tuning...")
    params = {
        "estimator__criterion": ["squared_error", "absolute_error"],
        "estimator__max_depth": [6, 7, 8, 9],
        "estimator__min_samples_split": [20, 30, 40],
        "estimator__min_samples_leaf": [10, 15, 20],
        "estimator__min_impurity_decrease": [0.001, 0.005, 0.01],
        "estimator__max_features": [0.8, 0.9, None],
        "estimator__ccp_alpha": [0.0, 0.001, 0.005]
    }

    base_dt = DecisionTreeRegressor(random_state=42)
    multi_dt = MultiOutputRegressor(base_dt)

    grid = GridSearchCV(multi_dt, params, cv=5, scoring="r2", n_jobs=-1, verbose=1)
    grid.fit(X_train_scaled, y_train)
    best_model = grid.best_estimator_

    print("\n🌟 Best Parameters:")
    for k, v in grid.best_params_.items():
        print(f"   {k}: {v}")

    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"\n📊 Cross-Validation R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 7. Retrain on combined data
    print("\n[6/9] Retraining on combined train+val...")
    X_combined = pd.concat([X_train_scaled, X_val_scaled])
    y_combined = pd.concat([y_train, y_val])
    best_model.fit(X_combined, y_combined)
    print("✅ Model retrained")

    # 8. Evaluate
    print("\n[7/9] Evaluating model...")

    def evaluate_model(model, X, y, name):
        y_pred = model.predict(X)
        # y and y_pred are (n_samples, n_targets)
        
        results = {}
        print(f"\n📊 {name.upper()}")
        print(f"{'='*70}")
        
        # Aggregate metrics
        mae_avg = mean_absolute_error(y, y_pred)
        rmse_avg = np.sqrt(mean_squared_error(y, y_pred))
        r2_avg = r2_score(y, y_pred)
        
        print(f"AVG  -> MAE: {mae_avg:.4f} | RMSE: {rmse_avg:.4f} | R²: {r2_avg:.4f}")
        
        # Per-target metrics
        y_np = y.values
        for i, col in enumerate(target_cols):
            y_true_col = y_np[:, i]
            y_pred_col = y_pred[:, i]
            
            mae = mean_absolute_error(y_true_col, y_pred_col)
            rmse = np.sqrt(mean_squared_error(y_true_col, y_pred_col))
            r2 = r2_score(y_true_col, y_pred_col)
            mape = np.mean(np.abs((y_true_col - y_pred_col) / (y_true_col + 1e-8))) * 100
            
            print(f"{col:<20} -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f} | MAPE: {mape:.2f}%")
            
            results[col] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

        results["AVG"] = {"MAE": mae_avg, "RMSE": rmse_avg, "R2": r2_avg}
        results["y_true"] = y_np
        results["y_pred"] = y_pred
        return results

    dt_train = evaluate_model(best_model, X_train_scaled, y_train, "DT - Train")
    dt_val = evaluate_model(best_model, X_val_scaled, y_val, "DT - Validation")
    dt_test = evaluate_model(best_model, X_test_scaled, y_test, "DT - Test")

    # 9. Overfitting check
    print("\n[8/9] Overfitting check (AVG R²)...")
    dt_gap = dt_train["AVG"]["R2"] - dt_test["AVG"]["R2"]
    print(f"Train R²: {dt_train['AVG']['R2']:.4f}")
    print(f"Test R²:  {dt_test['AVG']['R2']:.4f}")
    print(f"Gap:      {dt_gap:.4f} ({dt_gap*100:.2f}%)")

    # 10. Save results
    print("\n[9/9] Saving results...")
    
    # Save to Training folder
    joblib.dump(best_model, DT_DIR / "model.pkl")
    joblib.dump({
        'train': dt_train,
        'val': dt_val,
        'test': dt_test
    }, DT_DIR / "predictions.pkl")

    metrics = pd.DataFrame({
        'Dataset': ['Train', 'Validation', 'Test'],
        'MAE_AVG': [dt_train['AVG']['MAE'], dt_val['AVG']['MAE'], dt_test['AVG']['MAE']],
        'RMSE_AVG': [dt_train['AVG']['RMSE'], dt_val['AVG']['RMSE'], dt_test['AVG']['RMSE']],
        'R²_AVG': [dt_train['AVG']['R2'], dt_val['AVG']['R2'], dt_test['AVG']['R2']]
    })
    joblib.dump(metrics, DT_DIR / "metrics.pkl")

    # Feature importance (average over estimators if possible, or just base)
    # For MultiOutputRegressor, estimators_ is a list of estimators
    importances_list = []
    for i, est in enumerate(best_model.estimators_):
        importances_list.append(est.feature_importances_)
    
    avg_importance = np.mean(importances_list, axis=0)
    
    importances = pd.DataFrame(
        {'Feature': feature_cols,
         'Importance': avg_importance}
    ).sort_values('Importance', ascending=False)
    joblib.dump(importances, DT_DIR / "feature_importance.pkl")

    metadata = {
        'best_params': grid.best_params_,
        'cv_score': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
        'target_cols': target_cols
    }
    joblib.dump(metadata, DT_DIR / "metadata.pkl")
    joblib.dump(feature_cols, DT_DIR / "feature_cols.pkl")
    joblib.dump(target_cols, DT_DIR / "target_cols.pkl")
    
    # Save to models folder (untuk aplikasi)
    print("\n💾 Saving model to models folder...")
    joblib.dump(best_model, MODELS_DT_DIR / "model.pkl")
    joblib.dump(scaler, MODELS_DT_DIR / "scaler.pkl")
    joblib.dump(feature_cols, MODELS_DT_DIR / "feature_cols.pkl")
    joblib.dump(target_cols, MODELS_DT_DIR / "target_cols.pkl")
    joblib.dump(metadata, MODELS_DT_DIR / "metadata.pkl")
    joblib.dump(metrics, MODELS_DT_DIR / "metrics.pkl")
    joblib.dump(importances, MODELS_DT_DIR / "feature_importance.pkl")
    print(f"✅ Model saved to: {MODELS_DT_DIR}")

    print("\n✅ DECISION TREE TRAINING COMPLETE!")
    print(f"📁 Training artifacts: {DT_DIR}")
    print(f"📁 Models folder: {MODELS_DT_DIR}")
    print("="*80)


# ============================================================================
# CNN MULTI-HEAD TRAINING
# ============================================================================

def train_cnn():
    """Train CNN Multi-head model"""
    if not TENSORFLOW_AVAILABLE:
        print("\n⚠️ TensorFlow not available, skipping CNN training")
        return
    
    logger.info("Starting CNN Multi-head training")
    
    # Configuration
    GRID_SIZE = 15
    N_CHANNELS = 7
    N_POLLUTANTS = 3
    BATCH_SIZE = 16
    EPOCHS = 120
    
    CHANNEL_COLS = [
        'pm2_5 (μg/m³)_mean', 'ozone (μg/m³)_mean',
        'carbon_monoxide (μg/m³)_mean', 'temp',
        'humidity', 'windspeed', 'cloudcover'
    ]
    TARGET_COLS = ['pm2_5 (μg/m³)_mean', 'ozone (μg/m³)_mean', 'carbon_monoxide (μg/m³)_mean']
    POLLUTANT_NAMES = ["PM2.5", "Ozone", "CO"]
    
    OUT_DIR = CNN_DIR
    
    # Load data
    logger.info("Loading dataframes from %s", SHARED_DIR)
    try:
        # Try loading from CSV first
        train_df = pd.read_csv(SHARED_DIR / "train.csv")
        val_df = pd.read_csv(SHARED_DIR / "val.csv")
        test_df = pd.read_csv(SHARED_DIR / "test.csv")
        train_df['tanggal'] = pd.to_datetime(train_df['tanggal'], errors='coerce')
        val_df['tanggal'] = pd.to_datetime(val_df['tanggal'], errors='coerce')
        test_df['tanggal'] = pd.to_datetime(test_df['tanggal'], errors='coerce')
        logger.info("✅ Loaded from CSV")
    except FileNotFoundError as e:
        logger.error("❌ CSV files not found: %s", e)
        raise
    logger.info("Loaded: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))

    # Helpers
    def gaussian_source_map(value, grid_size=GRID_SIZE, sigma=3.0, center=(0.0,0.0)):
        x, y = np.meshgrid(np.linspace(-1,1,grid_size), np.linspace(-1,1,grid_size))
        cx, cy = center
        distance = np.sqrt((x-cx)**2 + (y-cy)**2)
        sigma_coord = max(1e-6, sigma / (grid_size / 2.0))
        kernel = np.exp(- (distance**2) / (2*(sigma_coord**2) + 1e-9))
        kernel = kernel / (kernel.sum() + 1e-9)
        return kernel * value

    def build_inputs_and_targets(df, add_spatial_variation=True):
        X_list, Yreg_list, Ymap_list = [], [], []
        for _, row in df.iterrows():
            grid = np.zeros((GRID_SIZE, GRID_SIZE, N_CHANNELS), dtype=float)
            for ch_idx, col in enumerate(CHANNEL_COLS):
                base = float(row[col]) if (col in row.index and pd.notna(row[col])) else 0.0
                if add_spatial_variation:
                    if col in ['pm2_5 (μg/m³)_mean', 'carbon_monoxide (μg/m³)_mean']:
                        center = (np.random.normal(0,0.25), np.random.normal(0,0.25))
                        kernel = gaussian_source_map(1.0, grid_size=GRID_SIZE, sigma=3.0, center=center)
                        var = kernel * base * 0.6
                    elif col == 'windspeed':
                        x = np.linspace(-1,1,GRID_SIZE)
                        wind_pattern = np.outer(x, np.ones(GRID_SIZE))
                        var = wind_pattern * base * 0.2
                    else:
                        var = np.random.normal(0, max(1e-6, abs(base)*0.12), (GRID_SIZE, GRID_SIZE))
                    grid[:,:,ch_idx] = np.maximum(base + var, 0.0)
                else:
                    grid[:,:,ch_idx] = np.maximum(base, 0.0)
            X_list.append(grid.astype(np.float32))

            # scalar reg target (raw)
            y_reg = np.array([float(row[t]) if (t in row.index and pd.notna(row[t])) else 0.0 for t in TARGET_COLS], dtype=float)
            Yreg_list.append(y_reg.astype(np.float32))

            # pixel-wise target map (generate spatialized map from scalar)
            y_map = np.zeros((GRID_SIZE, GRID_SIZE, N_POLLUTANTS), dtype=float)
            for p_idx, tcol in enumerate(TARGET_COLS):
                val = float(row[tcol]) if (tcol in row.index and pd.notna(row[tcol])) else 0.0
                num_sources = np.random.choice([1,1,2], p=[0.6,0.2,0.2])
                accum = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
                for _ in range(num_sources):
                    center = (np.random.normal(0,0.25), np.random.normal(0,0.25))
                    accum += gaussian_source_map(val / max(1, num_sources), grid_size=GRID_SIZE, sigma=3.0, center=center)
                y_map[:,:,p_idx] = np.maximum(accum, 0.0)
            Ymap_list.append(y_map.astype(np.float32))
        return np.array(X_list), np.array(Yreg_list), np.array(Ymap_list)

    # Build data
    logger.info("Building X,Y (this may take ~1-2 min)...")
    X_train, Yreg_train, Ymap_train = build_inputs_and_targets(train_df)
    X_val,   Yreg_val,   Ymap_val   = build_inputs_and_targets(val_df)
    X_test,  Yreg_test,  Ymap_test  = build_inputs_and_targets(test_df)
    logger.info("Shapes: X_train=%s, Yreg_train=%s, Ymap_train=%s", X_train.shape, Yreg_train.shape, Ymap_train.shape)

    # Scale X
    scaler_X = StandardScaler()
    X_train_flat = X_train.reshape(-1, N_CHANNELS)
    X_train_flat_s = scaler_X.fit_transform(X_train_flat)
    X_train = X_train_flat_s.reshape(X_train.shape)
    X_val = scaler_X.transform(X_val.reshape(-1, N_CHANNELS)).reshape(X_val.shape)
    X_test = scaler_X.transform(X_test.reshape(-1, N_CHANNELS)).reshape(X_test.shape)
    joblib.dump(scaler_X, OUT_DIR / "scaler_X.pkl")
    logger.info("Saved scaler_X")

    # Transform CO (log1p) for scalar targets
    Yreg_train_tf = Yreg_train.copy()
    Yreg_val_tf   = Yreg_val.copy()
    Yreg_test_tf  = Yreg_test.copy()

    Yreg_train_tf[:, 2] = np.log1p(Yreg_train_tf[:, 2])
    Yreg_val_tf[:, 2]   = np.log1p(Yreg_val_tf[:, 2])
    Yreg_test_tf[:, 2]  = np.log1p(Yreg_test_tf[:, 2])

    # Scale scalar targets with StandardScaler (fit on train-transformed)
    scaler_yreg = StandardScaler()
    Yreg_train_s = scaler_yreg.fit_transform(Yreg_train_tf)
    Yreg_val_s   = scaler_yreg.transform(Yreg_val_tf)
    Yreg_test_s  = scaler_yreg.transform(Yreg_test_tf)
    joblib.dump(scaler_yreg, OUT_DIR / "scaler_yreg.pkl")

    # Scale heatmap by per-channel max
    y_map_max = np.maximum(Ymap_train.max(axis=(0,1,2)), 1e-6)
    Ymap_train_s = Ymap_train / y_map_max.reshape((1,1,1,N_POLLUTANTS))
    Ymap_val_s   = Ymap_val   / y_map_max.reshape((1,1,1,N_POLLUTANTS))
    Ymap_test_s  = Ymap_test  / y_map_max.reshape((1,1,1,N_POLLUTANTS))
    joblib.dump(y_map_max, OUT_DIR / "y_map_max.pkl")

    # Model (multi-head)
    def build_multihead_cnn(grid_size=GRID_SIZE, in_channels=N_CHANNELS, out_channels=N_POLLUTANTS):
        """Enhanced CNN with encoder-decoder for heatmap + regression branch"""
        inp = keras.Input(shape=(grid_size, grid_size, in_channels), name="input_grid")

        # Encoder (reduced regularization untuk lebih banyak variasi)
        x = layers.Conv2D(32, 3, padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(0.0001))(inp)  # reduced from 0.001
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(0.0001))(x)  # reduced from 0.001
        x = layers.MaxPooling2D(2)(x)
        x = layers.SpatialDropout2D(0.1)(x)  # reduced from 0.2

        x = layers.Conv2D(128, 3, padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(0.00005))(x)  # reduced from 0.0001
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.SpatialDropout2D(0.05)(x)  # reduced from 0.1

        # Bottleneck
        x = layers.Conv2D(256, 3, padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(0.00001))(x)  # reduced from 0.00005
        x = layers.BatchNormalization()(x)

        # Heatmap decoder branch
        h = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        h = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(h)
        h = layers.BatchNormalization()(h)
        h = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(h)
        h = layers.BatchNormalization()(h)
        
        # ensure exact GRID_SIZE output even when intermediate dims are not exact
        h = layers.Resizing(grid_size, grid_size, interpolation='bilinear')(h)
        
        heatmap_out = layers.Conv2D(out_channels, 1, padding='same', activation='linear',  # linear activation untuk full range
                                    name='heatmap_out')(h)

        # Regression branch (scalar) - wider capacity
        r = layers.GlobalAveragePooling2D()(x)
        r = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.000001))(r)  # reduced from 0.00001
        r = layers.Dropout(0.1)(r)  # reduced from 0.15
        r = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.000001))(r)  # reduced from 0.00001
        r = layers.Dropout(0.05)(r)  # reduced from 0.1
        r = layers.Dense(64, activation='relu')(r)
        r = layers.Dense(32, activation='relu')(r)  # tambah layer untuk capacity
        reg_out = layers.Dense(out_channels, activation='linear', name='reg_out')(r)

        model = keras.Model(inp, outputs=[reg_out, heatmap_out], name="cnn_multihead")
        return model

    model = build_multihead_cnn()
    
    # Custom weighted regression loss with variance penalty (IMPROVES VARIATION)
    def weighted_reg_loss(y_true, y_pred):
        """Loss yang mendorong prediksi lebih bervariasi dengan variance penalty"""
        w = tf.constant([1.0, 1.0, 2.5], dtype=tf.float32)
        mae = tf.abs(y_true - y_pred) * w
        
        # Tambahkan penalty untuk prediksi yang terlalu uniform
        # Ini mendorong model untuk belajar lebih banyak variasi
        pred_std = tf.math.reduce_std(y_pred, axis=0)
        true_std = tf.math.reduce_std(y_true, axis=0)
        
        # Penalties untuk underfitting pada variasi
        variance_penalty = tf.reduce_mean(tf.maximum(0.0, true_std - pred_std)) * 0.1
        
        return tf.reduce_mean(mae) + variance_penalty

    opt = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=opt,
                  loss={'reg_out': weighted_reg_loss, 'heatmap_out': 'mse'},  # MSE lebih baik dr Huber untuk variasi
                  loss_weights={'reg_out':1.0, 'heatmap_out':0.8},  # kurangi heatmap weight
                  metrics={})
    model.summary()

    # Callbacks
    ckpt_path = OUT_DIR / "best_cnn_multihead.keras"
    mc = callbacks.ModelCheckpoint(str(ckpt_path), monitor='val_loss', save_best_only=True, verbose=1)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=False, verbose=1)
    rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, min_lr=1e-7, verbose=1)

    # Train
    history = model.fit(
        X_train, {'reg_out': Yreg_train_s, 'heatmap_out': Ymap_train_s},
        validation_data=(X_val, {'reg_out': Yreg_val_s, 'heatmap_out': Ymap_val_s}),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[mc, es, rlr], verbose=1
        )
    joblib.dump(history.history, OUT_DIR / "history.pkl")
    with open(OUT_DIR / "history.json","w") as fh:
        json.dump({k:[float(x) for x in v] for k,v in history.history.items()}, fh, indent=2)
    logger.info("Training finished and history saved")

    # Load best weights (if available)
    if ckpt_path.exists():
        try:
            model = keras.models.load_model(str(ckpt_path), compile=False)
            model.compile(optimizer=opt,
                          loss={'reg_out': weighted_reg_loss, 'heatmap_out': 'huber'},
                          loss_weights={'reg_out':1.0, 'heatmap_out':1.0},
                          metrics={})
            logger.info("Loaded model from checkpoint: %s", ckpt_path)
        except Exception as e:
            logger.warning("Could not load checkpoint model; using current weights. Error: %s", e)

    # Predict
    pred_reg_s, pred_map_s = model.predict(X_test, verbose=0)

    # Save the raw network outputs (normalized scalar preds and normalized maps)
    joblib.dump(pred_reg_s, OUT_DIR / "pred_reg_s_norm.pkl")
    joblib.dump(pred_map_s, OUT_DIR / "pred_map_s_norm.pkl")

    # inverse transform scalar predictions:
    pred_reg_tf = scaler_yreg.inverse_transform(pred_reg_s)   # this is on transformed (log1p for CO) scale
    # Save intermediate inverse-scaled predictions (before applying expm1)
    joblib.dump(pred_reg_tf, OUT_DIR / "pred_reg_tf_before_expm1.pkl")

    # 2) inverse log1p for CO (index 2)
    pred_reg = pred_reg_tf.copy()
    pred_reg[:,2] = np.expm1(pred_reg_tf[:,2])
    pred_reg = np.clip(pred_reg, 0.0, None)

    # inverse heatmap (rescale by y_map_max)
    pred_map = np.clip(pred_map_s, 0.0, 1.0) * y_map_max.reshape((1,1,1,N_POLLUTANTS))
    pred_map = np.clip(pred_map, 0.0, None)

    # save raw prediction artifacts (.npy and .pkl)
    np.save(OUT_DIR / "pred_reg_test.npy", pred_reg)
    np.save(OUT_DIR / "pred_map_test.npy", pred_map)
    np.save(OUT_DIR / "pred_map_test_norm.npy", pred_map_s)
    joblib.dump(pred_reg, OUT_DIR / "pred_reg_test.pkl")
    joblib.dump(pred_map, OUT_DIR / "pred_map_test.pkl")
    logger.info("Predictions saved to %s", OUT_DIR)

    # Diagnostics (CO & general)
    def stats_arr(a):
        return {
            "min": float(np.min(a)),
            "p1": float(np.percentile(a,1)),
            "p5": float(np.percentile(a,5)),
            "p25": float(np.percentile(a,25)),
            "p50": float(np.percentile(a,50)),
            "p75": float(np.percentile(a,75)),
            "p95": float(np.percentile(a,95)),
            "p99": float(np.percentile(a,99)),
            "max": float(np.max(a)),
            "mean": float(np.mean(a)),
            "std": float(np.std(a))
        }

    diag = {}
    # CO scalar ground truth (raw)
    diag['CO_ground_truth_raw_stats'] = stats_arr(Yreg_test[:,2])
    # pred after inverse-scaler (still in log1p domain)
    diag['CO_pred_after_inverse_scaler_stats (log1p domain)'] = stats_arr(pred_reg_tf[:,2])
    # pred after expm1 (final domain)
    diag['CO_pred_after_expm1_stats'] = stats_arr(pred_reg[:,2])

    # RMSE on log1p scale for CO (compare model's pred on transformed scale)
    y_true_log = np.log1p(Yreg_test[:,2])
    pred_log = pred_reg_tf[:,2]   # inverse-scaled values prior to expm1
    diag['CO_rmse_on_log1p_scale'] = float(math.sqrt(mean_squared_error(y_true_log, pred_log)))

    # NRMSE helper
    def nrmse(y_true, y_pred):
        denom = (np.max(y_true) - np.min(y_true)) if (np.max(y_true) - np.min(y_true)) > 0 else 1.0
        return float(np.sqrt(np.mean((y_true-y_pred)**2)) / denom)

    # scalar NRMSE for each pollutant
    diag['scalar_nrmse'] = {}
    for i,name in enumerate(POLLUTANT_NAMES):
        y_t = Yreg_test[:,i]
        y_p = pred_reg[:,i]
        diag['scalar_nrmse'][name] = nrmse(y_t, y_p)

    # RMSE & R2 quick checks (sanity) for scalars
    diag['scalar_basic'] = {}
    for i,name in enumerate(POLLUTANT_NAMES):
        y_t = Yreg_test[:,i]
        y_p = pred_reg[:,i]
        diag['scalar_basic'][name] = {
            'rmse': float(math.sqrt(mean_squared_error(y_t, y_p))),
            'r2': float(r2_score(y_t, y_p))
        }

    # Heatmap MAPE sensitivity across thresholds (for each pollutant)
    diag['heatmap_mape_sensitivity'] = {}
    factors = [0.01, 0.03, 0.05, 0.1, 0.2]
    for p,name in enumerate(POLLUTANT_NAMES):
        diag['heatmap_mape_sensitivity'][name] = {}
        for f in factors:
            thresh = f * float(y_map_max[p])
            # safe_mape function inline
            y_t = Ymap_test[...,p].ravel()
            y_p = pred_map[...,p].ravel()
            mask = y_t > thresh
            if mask.sum() == 0:
                mape_v = float('nan')
            else:
                mape_v = float(np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100.0)
            diag['heatmap_mape_sensitivity'][name][f] = {'thresh':float(thresh),'mape_percent':mape_v}

    # Helper: make JSON-serializable
    def make_serializable(obj):
        """Recursively convert numpy types/arrays to python native types for json.dump."""
        if isinstance(obj, dict):
            return {make_serializable(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(make_serializable(x) for x in obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if obj is None:
            return None
        # fallback for python builtin serializables
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

    # Save diagnostics (both joblib and JSON serializable)
    joblib.dump(diag, OUT_DIR / "diagnostics.pkl")
    diag_serial = make_serializable(diag)
    with open(OUT_DIR / "diagnostics.json","w") as fh:
        json.dump(diag_serial, fh, indent=2)
    logger.info("Diagnostics saved to %s", OUT_DIR / "diagnostics.json")

        # Also print key diagnostics to console
    logger.info("CO ground truth raw stats: %s", diag['CO_ground_truth_raw_stats'])
    logger.info("CO pred (after inverse scaler, log1p domain) stats: %s", diag['CO_pred_after_inverse_scaler_stats (log1p domain)'])
    logger.info("CO pred (after expm1) stats: %s", diag['CO_pred_after_expm1_stats'])
    logger.info("CO RMSE on log1p scale: %.6f", diag['CO_rmse_on_log1p_scale'])

    # Evaluation functions
    def rmse(y_true, y_pred):
        return math.sqrt(mean_squared_error(y_true, y_pred))

    def safe_mape_thresh(y_true, y_pred, thresh):
        y_t = np.array(y_true).ravel()
        y_p = np.array(y_pred).ravel()
        mask = y_t > thresh
        if mask.sum() == 0:
            return float('nan')
        return np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100.0

    # Evaluation (scalar)
    metrics = {}
    print("\n" + "="*70)
    print("REGRESSION METRICS (SCALAR) - TEST SET")
    print("="*70)
    metrics['regression_scalar'] = {}
    for i, name in enumerate(POLLUTANT_NAMES):
        y_t = Yreg_test[:, i]
        y_p = pred_reg[:, i]
        rmse_v = rmse(y_t, y_p)
        r2_v = r2_score(y_t, y_p)
        # MAPE threshold for scalars
        mape_v = safe_mape_thresh(y_t, y_p, thresh=1e-3)
        metrics['regression_scalar'][name] = {'rmse':rmse_v,'r2':r2_v,'mape_percent':mape_v}
        print(f"{name:<8} -> RMSE: {rmse_v:.4f} | R²: {r2_v:.4f} | MAPE: {mape_v if not math.isnan(mape_v) else 'nan':.2f}%")

    # Evaluation (heatmap pixel-wise)
    print("\n" + "="*70)
    print("REGRESSION METRICS (HEATMAP, PIXEL-WISE) - TEST SET")
    print("="*70)
    metrics['regression_heatmap'] = {}
    for p,name in enumerate(POLLUTANT_NAMES):
        y_t = Ymap_test[..., p].ravel()
        y_p = pred_map[..., p].ravel()
        rmse_v = rmse(y_t, y_p)
        r2_v = r2_score(y_t, y_p)
        thresh = 0.05 * float(y_map_max[p])
        mape_v = safe_mape_thresh(y_t, y_p, thresh=thresh)
        metrics['regression_heatmap'][name] = {'rmse':rmse_v,'r2':r2_v,'mape_percent':mape_v,'mape_thresh':thresh}
        print(f"{name:<8} -> RMSE: {rmse_v:.4f} | R²: {r2_v:.4f} | MAPE: {mape_v if not math.isnan(mape_v) else 'nan':.2f}% (thresh {thresh:.4f})")

    # Classification: worst pollutant pixel-wise
    print("\n" + "="*70)
    print("CLASSIFICATION (WORST POLLUTANT) - PIXEL-WISE")
    print("="*70)
    true_norm = (Ymap_test / (y_map_max.reshape((1,1,1,N_POLLUTANTS)) + 1e-12)).clip(0,1)
    pred_norm = (pred_map / (y_map_max.reshape((1,1,1,N_POLLUTANTS)) + 1e-12)).clip(0,1)

    true_worst = np.argmax(true_norm, axis=-1).ravel()
    pred_worst = np.argmax(pred_norm, axis=-1).ravel()

    f1_macro = f1_score(true_worst, pred_worst, average='macro', zero_division=0)
    f1_weighted = f1_score(true_worst, pred_worst, average='weighted', zero_division=0)
    prec_macro = precision_score(true_worst, pred_worst, average='macro', zero_division=0)
    rec_macro = recall_score(true_worst, pred_worst, average='macro', zero_division=0)
    cm = confusion_matrix(true_worst, pred_worst)

    metrics['classification'] = {
        'f1_macro': float(f1_macro), 'f1_weighted': float(f1_weighted),
        'precision_macro': float(prec_macro), 'recall_macro': float(rec_macro),
        'confusion_matrix': cm.tolist()
    }

    print(f"F1 (macro): {f1_macro:.4f} | F1 (weighted): {f1_weighted:.4f}")
    print(f"Precision (macro): {prec_macro:.4f} | Recall (macro): {rec_macro:.4f}")
    print("Confusion matrix (rows=true, cols=pred):\n", cm)
    print("\nPer-class classification report:\n")
    print(classification_report(true_worst, pred_worst, target_names=POLLUTANT_NAMES, zero_division=0))

    # Pixel MAE summary
    abs_err_map = np.mean(np.abs(Ymap_test - pred_map), axis=-1)  # (N,G,G)
    metrics['pixel_mae_mean'] = float(abs_err_map.mean())
    metrics['pixel_mae_std'] = float(abs_err_map.std())
    print("\nPixel-wise MAE map mean: %.4f | std: %.4f" % (metrics['pixel_mae_mean'], metrics['pixel_mae_std']))

    # Save metrics & diagnostics
    metrics_serial = make_serializable(metrics)
    with open(OUT_DIR / "evaluation_metrics_cnn_fixed.json", "w") as fh:
        json.dump(metrics_serial, fh, indent=2)
    joblib.dump(metrics, OUT_DIR / "evaluation_metrics_cnn_fixed.pkl")
    logger.info("Saved evaluation metrics to %s", OUT_DIR / "evaluation_metrics_cnn_fixed.json")

    # Save model & artifacts to Training folder
    try:
        model.save(OUT_DIR / "best_cnn.keras", include_optimizer=False)
    except Exception as e:
        logger.warning("Failed to save model in keras format: %s", e)
    joblib.dump(scaler_X, OUT_DIR / "scaler_X.pkl")
    joblib.dump(scaler_yreg, OUT_DIR / "scaler_yreg.pkl")
    joblib.dump(y_map_max, OUT_DIR / "y_map_max.pkl")

    logger.info("All done. Files saved to: %s", OUT_DIR)
    print("\nALL EVALUATION + DIAGNOSTICS COMPLETE. Metrics saved to:", OUT_DIR / "evaluation_metrics_cnn_fixed.json")
    
    # Save to models folder (untuk aplikasi)
    print("\n💾 Saving model to models folder...")
    try:
        model.save(MODELS_CNN_DIR / "best_cnn.keras", include_optimizer=False)
        print(f"✅ Model saved to: {MODELS_CNN_DIR / 'best_cnn.keras'}")
    except Exception as e:
        logger.warning("Failed to save model to models folder: %s", e)
    joblib.dump(scaler_X, MODELS_CNN_DIR / "scaler_X.pkl")
    joblib.dump(scaler_yreg, MODELS_CNN_DIR / "scaler_yreg.pkl")
    joblib.dump(y_map_max, MODELS_CNN_DIR / "y_map_max.pkl")
    joblib.dump(metrics, MODELS_CNN_DIR / "evaluation_metrics_cnn_fixed.pkl")
    with open(MODELS_CNN_DIR / "evaluation_metrics_cnn_fixed.json", "w") as fh:
        json.dump(metrics_serial, fh, indent=2)
    print(f"✅ All CNN artifacts saved to: {MODELS_CNN_DIR}")


# ============================================================================
# GRU MULTI-OUTPUT TRAINING
# ============================================================================

def train_gru():
    """Train GRU Multi-output model"""
    if not TENSORFLOW_AVAILABLE:
        print("\n⚠️ TensorFlow not available, skipping GRU training")
        return
    
    print("MULTI-OUTPUT GRU (Ultra Minimal) — pm2.5, ozone, CO")
    print("=" * 80)
    
    OUT_DIR = GRU_DIR
    
    BATCH_SIZE = 16
    SEQUENCE_LENGTH = 1
    EPOCHS = 250
    
    # Load data
    try:
        # Try loading from CSV first
        train_df = pd.read_csv(SHARED_DIR / "train.csv")
        val_df = pd.read_csv(SHARED_DIR / "val.csv")
        test_df = pd.read_csv(SHARED_DIR / "test.csv")
        train_df['tanggal'] = pd.to_datetime(train_df['tanggal'], errors='coerce')
        val_df['tanggal'] = pd.to_datetime(val_df['tanggal'], errors='coerce')
        test_df['tanggal'] = pd.to_datetime(test_df['tanggal'], errors='coerce')
        print("✅ Loaded from CSV")
    except FileNotFoundError as e:
        print(f"❌ CSV files not found: {e}")
        raise

    # Identify target columns robustly
    def find_target_column(df, keywords):
        for kw in keywords:
            matches = [c for c in df.columns if kw in c.lower() and 'mean' in c.lower()]
            if matches:
                return matches[0]
        for kw in keywords:
            matches = [c for c in df.columns if kw in c.lower()]
            if matches:
                return matches[0]
        return None

    t_pm25 = find_target_column(train_df, ['pm2_5', 'pm2.5', 'pm2'])
    t_o3   = find_target_column(train_df, ['ozone', 'o3'])
    t_co   = find_target_column(train_df, ['carbon_monoxide', 'carbon monoxide', 'co'])

    if not (t_pm25 and t_o3 and t_co):
        raise RuntimeError("Could not detect all target columns. Detected:"
                           f" pm2.5={t_pm25}, o3={t_o3}, co={t_co}")

    target_cols = [t_pm25, t_o3, t_co]
    print("Detected targets:", target_cols)

    # Optional feature engineering
    def create_temporal_features(df, targets):
        df = df.copy()
        if 'tanggal' in df.columns:
            df['tanggal'] = pd.to_datetime(df['tanggal'])
            df = df.sort_values('tanggal').reset_index(drop=True)

        for t in targets:
            for lag in (1, 2, 3, 7):
                df[f"{t}_lag_{lag}"] = df[t].shift(lag)
            df[f"{t}_roll_mean_7"] = df[t].rolling(7, min_periods=1).mean()
            df[f"{t}_roll_std_7"]  = df[t].rolling(7, min_periods=1).std()
            df[f"{t}_diff_1"] = df[t].diff(1)

        if 'tanggal' in df.columns:
            df['day'] = df['tanggal'].dt.day
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 30)
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 30)

        created = [c for c in df.columns if any(s in c for s in ['_lag_', 'roll_mean', 'roll_std', '_diff_'])]
        df = df.dropna(subset=created).reset_index(drop=True)
        return df

    # Apply temporal features
    train_df = create_temporal_features(train_df, target_cols)
    val_df   = create_temporal_features(val_df, target_cols)
    test_df  = create_temporal_features(test_df, target_cols)

    # Feature columns
    exclude_cols = ['tanggal', 'tanggal_waktu']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols + target_cols]
    print(f"Using {len(feature_cols)} features.")

    # CO winsorize + log1p (train thresholds)
    co_vals_train = train_df[t_co].dropna().values
    p_low, p_high = np.percentile(co_vals_train, [1, 99])
    print("CO winsorize thresholds (train 1%,99%) =", p_low, p_high)

    def winsorize(arr, low, high):
        return np.clip(arr, low, high)

    train_co_w = winsorize(train_df[t_co].values, p_low, p_high)
    val_co_w   = winsorize(val_df[t_co].values, p_low, p_high)
    test_co_w  = winsorize(test_df[t_co].values, p_low, p_high)

    shift = 0.0
    min_train = train_co_w.min()
    if min_train <= 0:
        shift = abs(min_train) + 1e-6
        print("Shifting CO by", shift)

    train_co_log = np.log1p(train_co_w + shift)
    val_co_log   = np.log1p(val_co_w + shift)
    test_co_log  = np.log1p(test_co_w + shift)

    # Build Y dataframes where CO is transformed
    y_train_df = train_df[[t_pm25, t_o3]].copy()
    y_train_df[t_co] = train_co_log

    y_val_df = val_df[[t_pm25, t_o3]].copy()
    y_val_df[t_co] = val_co_log

    y_test_df = test_df[[t_pm25, t_o3]].copy()
    y_test_df[t_co] = test_co_log

    # Scaling: X single scaler, Y per-target scalers
    scaler_X = RobustScaler()
    X_train_scaled = scaler_X.fit_transform(train_df[feature_cols])
    X_val_scaled   = scaler_X.transform(val_df[feature_cols])
    X_test_scaled  = scaler_X.transform(test_df[feature_cols])
    joblib.dump(scaler_X, OUT_DIR / "scaler_X.pkl")

    scalers_y = {}
    y_train_scaled = np.zeros((len(y_train_df), len(target_cols)))
    y_val_scaled   = np.zeros((len(y_val_df), len(target_cols)))
    y_test_scaled  = np.zeros((len(y_test_df), len(target_cols)))

    from sklearn.preprocessing import RobustScaler as _Robust
    for i, col in enumerate(target_cols):
        sc = _Robust()
        y_train_scaled[:, i] = sc.fit_transform(y_train_df[[col]]).flatten()
        y_val_scaled[:, i]   = sc.transform(y_val_df[[col]]).flatten()
        y_test_scaled[:, i]  = sc.transform(y_test_df[[col]]).flatten()
        scalers_y[col] = sc

    joblib.dump(scalers_y, OUT_DIR / "scalers_y.pkl")
    joblib.dump(feature_cols, OUT_DIR / "features.pkl")
    joblib.dump(target_cols, OUT_DIR / "targets.pkl")
    print("Saved scalers & metadata.")

    # Sequence creation
    def create_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len])
        return np.array(Xs), np.array(ys)

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
    X_val_seq,   y_val_seq   = create_sequences(X_val_scaled,   y_val_scaled,   SEQUENCE_LENGTH)
    X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test_scaled,  SEQUENCE_LENGTH)

    print("Sequence shapes:")
    print(" X_train:", X_train_seq.shape, "y_train:", y_train_seq.shape)
    print(" X_val:  ", X_val_seq.shape,   "y_val:  ", y_val_seq.shape)
    print(" X_test: ", X_test_seq.shape,  "y_test: ", y_test_seq.shape)

    if X_train_seq.shape[0] == 0:
        raise RuntimeError("Not enough sequences. Reduce SEQUENCE_LENGTH or add data.")

    # Build multi-head GRU (enhanced for better variation)
    def build_simple_model(input_shape, n_targets, lr=5e-4):
        """
        Enhanced GRU model with better capacity for capturing variations:
        - Dual GRU layers (64 + 32 units) untuk lebih banyak capacity
        - Reduced regularization untuk eksplorasi penuh output space
        - Variance-aware loss
        """
        inp = keras.Input(shape=input_shape)

        # Dual GRU layers dengan lebih banyak units
        x = layers.GRU(64,  # increased from 8
                       return_sequences=True,
                       kernel_regularizer=regularizers.l2(0.0001),  # reduced from 0.01
                       recurrent_regularizer=regularizers.l2(0.00005))(inp)
        x = layers.Dropout(0.1)(x)
        
        x = layers.GRU(32,  # tambah layer kedua
                       return_sequences=False,
                       kernel_regularizer=regularizers.l2(0.0001),
                       recurrent_regularizer=regularizers.l2(0.00005))(x)
        x = layers.Dropout(0.1)(x)

        # Dense layers dengan lebih banyak capacity
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(x)
        x = layers.Dropout(0.08)(x)
        x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(x)
        x = layers.Dropout(0.05)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Multi-head outputs (no shared layer sebelum output untuk independence)
        outputs = []
        for i in range(n_targets):
            out = layers.Dense(1, activation='linear', name=f'out_{i}')(x)  # linear untuk full range
            outputs.append(out)

        model = keras.Model(inputs=inp, outputs=outputs)

        # Loss dengan variance penalty
        loss_dict = {}
        metrics_dict = {}
        for i in range(n_targets):
            loss_dict[f'out_{i}'] = 'mse'
            metrics_dict[f'out_{i}'] = ['mae']

        optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=loss_dict, metrics=metrics_dict)
        return model

    model = build_simple_model((SEQUENCE_LENGTH, X_train_seq.shape[2]), len(target_cols), lr=5e-4)
    model.summary()

    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1),
        callbacks.ModelCheckpoint(str(OUT_DIR / "best_multi_conservative.weights.h5"),
                                  monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
    ]

    # Prepare y list for multi-head training
    y_train_list = [y_train_seq[:, i] for i in range(len(target_cols))]
    y_val_list   = [y_val_seq[:, i]   for i in range(len(target_cols))]

    # Train
    history = model.fit(
        X_train_seq,
        y_train_list,
        validation_data=(X_val_seq, y_val_list),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list,
        shuffle=False,
        verbose=1
    )

    # Evaluation helpers
    def inverse_preds(preds_list):
        if isinstance(preds_list, list):
            pred_scaled = np.column_stack([p.flatten() for p in preds_list])
        else:
            pred_scaled = preds_list
        pred_orig = np.zeros_like(pred_scaled, dtype=float)
        for i, col in enumerate(target_cols):
            sc = scalers_y[col]
            pred_orig[:, i] = sc.inverse_transform(pred_scaled[:, i].reshape(-1,1)).flatten()
        # CO: inverse log1p + shift (add shift back, not subtract)
        co_idx = target_cols.index(t_co)
        pred_orig[:, co_idx] = np.expm1(pred_orig[:, co_idx]) + shift
        return pred_orig

    def inverse_true(y_scaled_arr):
        true_orig = np.zeros_like(y_scaled_arr, dtype=float)
        for i, col in enumerate(target_cols):
            sc = scalers_y[col]
            true_orig[:, i] = sc.inverse_transform(y_scaled_arr[:, i].reshape(-1,1)).flatten()
        co_idx = target_cols.index(t_co)
        true_orig[:, co_idx] = np.expm1(true_orig[:, co_idx]) + shift
        return true_orig

    def evaluate_model(model, X_seq, y_scaled, set_name):
        preds = model.predict(X_seq, verbose=0)
        y_pred_orig = inverse_preds(preds)
        y_true_orig = inverse_true(y_scaled)

        print(f"\n📊 {set_name} SET")
        print("=" * 70)
        for i, col in enumerate(target_cols):
            mae = mean_absolute_error(y_true_orig[:, i], y_pred_orig[:, i])
            rmse = np.sqrt(mean_squared_error(y_true_orig[:, i], y_pred_orig[:, i]))
            r2 = r2_score(y_true_orig[:, i], y_pred_orig[:, i])
            print(f"{col} -- MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
        agg_r2 = np.mean([r2_score(y_true_orig[:, i], y_pred_orig[:, i]) for i in range(len(target_cols))])
        print(f"\nAggregated mean R²: {agg_r2:.4f}")
        return y_true_orig, y_pred_orig

    # Run evaluation
    train_true, train_pred = evaluate_model(model, X_train_seq, y_train_seq, "TRAIN")
    val_true,   val_pred   = evaluate_model(model, X_val_seq,   y_val_seq,   "VALIDATION")
    test_true,  test_pred  = evaluate_model(model, X_test_seq,  y_test_seq,  "TEST")

    # Overfitting check
    train_r2_mean = np.mean([r2_score(train_true[:, i], train_pred[:, i]) for i in range(len(target_cols))])
    test_r2_mean  = np.mean([r2_score(test_true[:, i], test_pred[:, i]) for i in range(len(target_cols))])
    gap = train_r2_mean - test_r2_mean
    print(f"\n🔍 Overfitting check: Train mean R²={train_r2_mean:.4f}, Test mean R²={test_r2_mean:.4f}, Gap={gap:.4f}")
    
    # Debug output: Show sample test predictions
    print("\n🔧 DEBUG: Sample test predictions (first 10 rows)")
    print("Target columns:", target_cols)
    print("NOTE: CO values are after inverse log1p + shift (should be in original scale ~300-1000)")
    for i, col in enumerate(target_cols):
        print(f"\n{col}:")
        print(f"  True  (first 10): {test_true[:10, i]}")
        print(f"  Pred  (first 10): {test_pred[:10, i]}")
        print(f"  Min True: {test_true[:, i].min():.4f}, Max True: {test_true[:, i].max():.4f}")
        print(f"  Min Pred: {test_pred[:, i].min():.4f}, Max Pred: {test_pred[:, i].max():.4f}")

    # Save model & artifacts to Training folder (use .keras format)
    best_path = OUT_DIR / "best_multi_conservative.weights.h5"
    if best_path.exists():
        model.load_weights(str(best_path))
        print("Loaded best weights from:", best_path)
    else:
        print("Best weights not found. Using current weights.")

    # Save Keras native format (.keras) to Training folder
    model_path = OUT_DIR / "model_v9.keras"
    model.save(str(model_path), include_optimizer=False)
    print("Saved Keras model to:", model_path)

    # Save scalers and metadata to Training folder
    joblib.dump(scalers_y, OUT_DIR / "scalers_y.pkl")
    joblib.dump(scaler_X, OUT_DIR / "scaler_X.pkl")
    joblib.dump(feature_cols, OUT_DIR / "features.pkl")
    joblib.dump(target_cols, OUT_DIR / "targets.pkl")
    print("Saved scalers to:", OUT_DIR)

    # Save test predictions CSV for inspection
    pred_df = pd.DataFrame(test_pred, columns=[f"pred_{c}" for c in target_cols])
    true_df = pd.DataFrame(test_true, columns=[f"true_{c}" for c in target_cols])
    out_df = pd.concat([true_df, pred_df], axis=1)
    out_df.to_csv(OUT_DIR / "multi_test_predictions_conservative.csv", index=False)
    print("Saved test predictions to:", OUT_DIR / "multi_test_predictions_conservative.csv")
    
    # Save to models folder (untuk aplikasi)
    print("\n💾 Saving model to models folder...")
    model.save(MODELS_GRU_DIR / "model_v9.keras", include_optimizer=False)
    print(f"✅ Model saved to: {MODELS_GRU_DIR / 'model_v9.keras'}")
    joblib.dump(scalers_y, MODELS_GRU_DIR / "scalers_y.pkl")
    joblib.dump(scaler_X, MODELS_GRU_DIR / "scaler_X.pkl")
    joblib.dump(feature_cols, MODELS_GRU_DIR / "features.pkl")
    joblib.dump(target_cols, MODELS_GRU_DIR / "targets.pkl")
    print(f"✅ All GRU artifacts saved to: {MODELS_GRU_DIR}")

    print("\nMULTI-OUTPUT GRU (conservative) RUN COMPLETE")
    print(f"📁 Training artifacts: {OUT_DIR}")
    print(f"📁 Models folder: {MODELS_GRU_DIR}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Air Quality Prediction Models')
    parser.add_argument('--preprocess', action='store_true', help='Run preprocessing')
    parser.add_argument('--decision-tree', action='store_true', help='Train Decision Tree')
    parser.add_argument('--gru', action='store_true', help='Train GRU model (requires TensorFlow)')
    parser.add_argument('--cnn', action='store_true', help='Train CNN model (requires TensorFlow)')
    parser.add_argument('--all', action='store_true', help='Run all training steps')
    
    args = parser.parse_args()
    
    # If no arguments, run all
    if not any(vars(args).values()):
        args.all = True
    
    try:
        if args.preprocess or args.all:
            preprocess_data()
        
        if args.decision_tree or args.all:
            train_decision_tree()
        
        if args.gru or args.all:
            train_gru()
        
        if args.cnn or args.all:
            train_cnn()
        
        print("\n✅ Training pipeline completed!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()