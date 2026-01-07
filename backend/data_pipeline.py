#!/usr/bin/env python3
"""
IMPLEMENTATION: Continuous Learning Data Flow
Dengan 2-Layer Dataset Structure:
  1. aggregate_daily.csv (RAW) - Growing dataset
  2. dataset_preprocessed.csv - Preprocessed untuk training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent  # backend dir
DATA_DIR = BASE_DIR / "data"
PREPROCESSED_DIR = DATA_DIR / "dataset_preprocessed"
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Two-layer dataset
RAW_DATASET_PATH = PREPROCESSED_DIR / "aggregate_daily.csv"  # Layer 1: RAW
PROCESSED_DATASET_PATH = PREPROCESSED_DIR / "dataset_preprocessed.csv"  # Layer 2: PROCESSED

# ============================================================================
# LAYER 1: RAW DATASET MANAGEMENT
# ============================================================================

def create_or_load_raw_dataset():
    """
    Create/load aggregate_daily.csv (Layer 1 RAW).
    
    Ini file yang GROW setiap hari:
    - Day 1: 200 rows (historical)
    - Day 2: 201 rows (200 + 1 baru)
    - Day 30: 230 rows (200 + 30 baru)
    
    Columns (15):
      tanggal, pm2_5_mean, pm2_5_max, pm2_5_median, 
      ozone_mean, ozone_max, ozone_median,
      co_mean, co_max, co_median,
      temp, humidity, windspeed, pressure, cloudcover, visibility, solar
    """
    
    if RAW_DATASET_PATH.exists():
        df = pd.read_csv(RAW_DATASET_PATH)
        logger.info(f"[RAW] Loaded existing raw dataset: {len(df)} rows")
        return df
    else:
        logger.warning(f"[RAW] Raw dataset not found, creating empty")
        # Nanti akan di-append dengan data baru
        return pd.DataFrame()


def append_aggregation_to_raw_dataset(aggregation: dict):
    """
    STEP 2 of Daily Training Pipeline.
    
    Append hasil agregasi ke Layer 1 (RAW dataset).
    Input: aggregation dict dari aggregate_hourly_to_daily()
    Output: aggregate_daily.csv dengan 1 row ditambah
    """
    
    try:
        if aggregation is None:
            logger.warning("[RAW] Aggregation is None, skip append")
            return False
        
        target_date = aggregation.get('tanggal')
        
        # Load existing raw dataset
        df_raw = create_or_load_raw_dataset()
        
        # Check if date already exists
        if len(df_raw) > 0 and 'tanggal' in df_raw.columns:
            existing = df_raw[df_raw['tanggal'] == target_date]
            if len(existing) > 0:
                logger.warning(f"[RAW] Data {target_date} sudah ada, skip")
                return False
        
        # Prepare row untuk append
        row = {
            'tanggal': target_date,
            'pm2_5_mean': aggregation.get('pm2_5 (μg/m³)_mean'),
            'pm2_5_max': aggregation.get('pm2_5 (μg/m³)_max'),
            'pm2_5_median': aggregation.get('pm2_5 (μg/m³)_median'),
            'ozone_mean': aggregation.get('ozone (μg/m³)_mean'),
            'ozone_max': aggregation.get('ozone (μg/m³)_max'),
            'ozone_median': aggregation.get('ozone (μg/m³)_median'),
            'co_mean': aggregation.get('carbon_monoxide (μg/m³)_mean'),
            'co_max': aggregation.get('carbon_monoxide (μg/m³)_max'),
            'co_median': aggregation.get('carbon_monoxide (μg/m³)_median'),
            'temp': aggregation.get('temp'),
            'tempmax': aggregation.get('tempmax'),
            'tempmin': aggregation.get('tempmin'),
            'humidity': aggregation.get('humidity'),
            'windspeed': aggregation.get('windspeed'),
            'winddir': aggregation.get('winddir'),
            'sealevelpressure': aggregation.get('sealevelpressure'),
            'cloudcover': aggregation.get('cloudcover'),
            'visibility': aggregation.get('visibility'),
            'solarradiation': aggregation.get('solarradiation'),
        }
        
        # Append
        df_new = pd.concat([df_raw, pd.DataFrame([row])], ignore_index=True)
        
        # Remove duplicates & sort
        df_new = df_new.drop_duplicates(subset=['tanggal'], keep='first')
        df_new = df_new.sort_values('tanggal', ascending=True).reset_index(drop=True)
        
        # Save
        df_new.to_csv(RAW_DATASET_PATH, index=False)
        
        logger.info(f"[RAW] ✅ Appended {target_date}")
        logger.info(f"   Previous: {len(df_raw)} rows")
        logger.info(f"   New: {len(df_new)} rows")
        
        return True
        
    except Exception as e:
        logger.error(f"[RAW] Error appending to raw dataset: {e}", exc_info=True)
        return False


# ============================================================================
# LAYER 2: PREPROCESSING PIPELINE (RE-PROCESS SEMUA DATA)
# ============================================================================

def preprocess_raw_to_preprocessed():
    """
    STEP 3 of Daily Training Pipeline.
    
    RE-PREPROCESS SEMUA data dari raw ke preprocessed.
    Input: aggregate_daily.csv (201 rows × 15 features)
    Output: dataset_preprocessed.csv (201 rows × 50+ features)
    
    Pipeline:
      1. Outlier Detection (Z-score)
      2. Missing Value Imputation
      3. Feature Engineering (temporal, interactions, rolling)
      4. Scaling (MinMax)
      5. Save dengan version tracking
    """
    
    try:
        logger.info("[PREPROCESS] Starting preprocessing pipeline...")
        
        # Load Layer 1 (RAW)
        df = pd.read_csv(RAW_DATASET_PATH)
        initial_rows = len(df)
        logger.info(f"[PREPROCESS] Loaded {initial_rows} rows dari raw dataset")
        
        if initial_rows == 0:
            logger.warning("[PREPROCESS] Raw dataset kosong!")
            return False
        
        # Convert tanggal to datetime
        df['tanggal'] = pd.to_datetime(df['tanggal'])
        
        # ===== STEP 1: OUTLIER DETECTION =====
        logger.info("[PREPROCESS] STEP 1: Outlier Detection")
        
        # Z-score thresholding
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in ['pm2_5_mean', 'ozone_mean', 'co_mean']:
            if col in df.columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                # Mark outliers (Z > 3) tapi jangan delete, mark sebagai NaN
                df.loc[z_scores > 3, col] = np.nan
                logger.info(f"   {col}: Detected outliers")
        
        # ===== STEP 2: MISSING VALUE IMPUTATION =====
        logger.info("[PREPROCESS] STEP 2: Missing Value Imputation")
        
        # Linear interpolation
        df = df.interpolate(method='linear', limit_direction='both')
        
        # Backward fill untuk missing di awal/akhir
        df = df.bfill().ffill()
        
        logger.info(f"   After imputation: {df.isnull().sum().sum()} nulls remaining")
        
        # ===== STEP 3: FEATURE ENGINEERING =====
        logger.info("[PREPROCESS] STEP 3: Feature Engineering")
        
        # 3.1: Temporal Features
        df['day_of_year'] = df['tanggal'].dt.dayofyear
        df['month'] = df['tanggal'].dt.month
        df['quarter'] = df['tanggal'].dt.quarter
        df['is_weekend'] = df['tanggal'].dt.dayofweek.isin([5, 6]).astype(int)
        df['week_of_year'] = df['tanggal'].dt.isocalendar().week
        
        logger.info("   Created temporal features: day_of_year, month, quarter, is_weekend, week_of_year")
        
        # 3.2: Trend Features (PM2.5 trend over time)
        df['pm25_trend'] = df['pm2_5_mean'].diff().fillna(0)
        logger.info("   Created trend features: pm25_trend")
        
        # 3.3: Interaction Features
        df['pm25_humidity'] = df['pm2_5_mean'] * df['humidity']
        df['pm25_temp'] = df['pm2_5_mean'] * df['temp']
        df['ozone_temp'] = df['ozone_mean'] * df['temp']
        df['humidity_temp'] = df['humidity'] * df['temp']
        logger.info("   Created interaction features: pm25_humidity, pm25_temp, ozone_temp, humidity_temp")
        
        # 3.4: Rolling Aggregation Features (untuk continuous learning, window kecil dulu)
        # Window size tergantung jumlah data
        if initial_rows >= 30:
            # 7-day rolling
            df['pm25_7d_mean'] = df['pm2_5_mean'].rolling(window=7, center=False).mean()
            df['pm25_7d_std'] = df['pm2_5_mean'].rolling(window=7, center=False).std()
            df['pm25_7d_max'] = df['pm2_5_mean'].rolling(window=7, center=False).max()
            df['pm25_7d_min'] = df['pm2_5_mean'].rolling(window=7, center=False).min()
            
            df['ozone_7d_mean'] = df['ozone_mean'].rolling(window=7, center=False).mean()
            df['co_7d_mean'] = df['co_mean'].rolling(window=7, center=False).mean()
            
            logger.info("   Created 7-day rolling features")
        
        if initial_rows >= 60:
            # 30-day rolling
            df['pm25_30d_mean'] = df['pm2_5_mean'].rolling(window=30, center=False).mean()
            df['pm25_30d_std'] = df['pm2_5_mean'].rolling(window=30, center=False).std()
            
            logger.info("   Created 30-day rolling features")
        
        # Fill NaN dari rolling windows
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # 3.5: Lag Features (untuk temporal dependence)
        if initial_rows >= 7:
            df['pm25_lag1'] = df['pm2_5_mean'].shift(1)
            df['pm25_lag3'] = df['pm2_5_mean'].shift(3)
            df['pm25_lag7'] = df['pm2_5_mean'].shift(7)
            
            logger.info("   Created lag features: pm25_lag1, lag3, lag7")
        
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # ===== STEP 4: SCALING (NORMALIZATION) =====
        logger.info("[PREPROCESS] STEP 4: Scaling/Normalization")
        
        from sklearn.preprocessing import MinMaxScaler
        import joblib
        
        # Select numeric columns untuk scaling
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['day_of_year', 'month', 'quarter', 
                                                               'is_weekend', 'week_of_year']]
        
        # Fit scaler dengan data terbaru (201 rows)
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        logger.info(f"   Scaled {len(numeric_cols)} numeric columns to [0, 1]")
        
        # Save scaler untuk deployment
        scaler_path = PREPROCESSED_DIR / 'minmax_scaler.pkl'
        joblib.dump(scaler, scaler_path)
        logger.info(f"   Saved scaler: {scaler_path}")
        
        # ===== STEP 5: SAVE PREPROCESSED DATASET =====
        logger.info("[PREPROCESS] STEP 5: Save Preprocessed Dataset")
        
        # Drop tanggal temporarily (save as string)
        df['tanggal'] = df['tanggal'].dt.strftime('%Y-%m-%d')
        
        df.to_csv(PROCESSED_DATASET_PATH, index=False)
        
        final_rows = len(df)
        final_cols = len(df.columns)
        
        logger.info(f"[PREPROCESS] ✅ Preprocessing complete!")
        logger.info(f"   Input: {initial_rows} rows × 15 columns (raw)")
        logger.info(f"   Output: {final_rows} rows × {final_cols} columns (preprocessed)")
        logger.info(f"   Saved: {PROCESSED_DATASET_PATH}")
        
        return True
        
    except Exception as e:
        logger.error(f"[PREPROCESS] Error during preprocessing: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# LAYER 2.5: LOAD PREPROCESSED DATA FOR TRAINING
# ============================================================================

def load_preprocessed_for_training():
    """
    Load Layer 2 (PREPROCESSED) untuk training models.
    
    Return: (X, y, feature_columns)
    """
    
    try:
        logger.info("[LOAD] Loading preprocessed dataset for training...")
        
        df = pd.read_csv(PROCESSED_DATASET_PATH)
        
        logger.info(f"[LOAD] Loaded {len(df)} rows × {len(df.columns)} columns")
        
        # Assume kolom target adalah pollutant measurements (before scaling)
        # Untuk training, kita gunakan data yang sudah scaled
        
        # Feature columns (exclude tanggal dan datetime)
        exclude_cols = ['tanggal', 'datetime']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols].values
        
        # Targets (PM2.5, O3, CO) - assuming available in preprocessed
        targets = ['pm2_5_mean', 'ozone_mean', 'co_mean']
        target_cols = [c for c in targets if c in df.columns]
        
        if not target_cols:
            logger.error("[LOAD] Target columns not found!")
            return None, None, feature_cols
        
        y = df[target_cols].values
        
        logger.info(f"[LOAD] X shape: {X.shape}")
        logger.info(f"[LOAD] y shape: {y.shape}")
        logger.info(f"[LOAD] Features: {len(feature_cols)}")
        
        return X, y, feature_cols
        
    except Exception as e:
        logger.error(f"[LOAD] Error loading preprocessed data: {e}")
        return None, None, None


# ============================================================================
# SUMMARY & USAGE
# ============================================================================

"""
USAGE IN SCHEDULER:

def daily_aggregation_and_training_job():
    logger.info("[DAILY] Starting daily aggregation and training...")
    
    # [STEP 1] Aggregation
    from crawler.db_handler import aggregate_hourly_to_daily
    aggregation = aggregate_hourly_to_daily(target_date=date.today())
    
    # [STEP 2] Merge to RAW dataset (Layer 1)
    from data_pipeline import append_aggregation_to_raw_dataset
    success = append_aggregation_to_raw_dataset(aggregation)
    
    # [STEP 3] Preprocess SEMUA data (Layer 1 → Layer 2)
    from data_pipeline import preprocess_raw_to_preprocessed
    success = preprocess_raw_to_preprocessed()
    
    # [STEP 4] Load & Train
    from data_pipeline import load_preprocessed_for_training
    X, y, features = load_preprocessed_for_training()
    
    # [STEP 5] Train models
    from training import train_decision_tree, train_cnn, train_gru
    dt_model = train_decision_tree(X, y)
    cnn_model = train_cnn(X, y)
    gru_model = train_gru(X, y)
    
    # [STEP 6] Save models
    # ... save logic ...
    
    logger.info("[DAILY] ✅ Training cycle complete!")
"""

if __name__ == "__main__":
    # Test preprocessing pipeline
    print("\n" + "="*70)
    print("TEST: CONTINUOUS LEARNING DATA PIPELINE")
    print("="*70 + "\n")
    
    # Konfigure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("[1] Loading raw dataset...")
    df_raw = create_or_load_raw_dataset()
    print(f"    Status: {len(df_raw)} rows\n")
    
    print("[2] Preprocessing pipeline...")
    success = preprocess_raw_to_preprocessed()
    print(f"    Status: {'✅ Success' if success else '❌ Failed'}\n")
    
    print("[3] Loading for training...")
    X, y, features = load_preprocessed_for_training()
    if X is not None:
        print(f"    X shape: {X.shape}")
        print(f"    y shape: {y.shape}")
        print(f"    Features: {len(features)}\n")
    
    print("="*70)
    print("✅ Data pipeline test complete!")
    print("="*70)
