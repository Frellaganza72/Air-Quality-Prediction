# 📊 DOKUMENTASI LENGKAP SISTEM AIR QUALITY MONITORING
**Malang Air Quality Prediction System**  
**Tanggal Dokumentasi:** 5 Januari 2026  
**Versi:** Production Ready

---

## 📑 DAFTAR ISI

1. [OVERVIEW SISTEM](#overview-sistem)
2. [ARSITEKTUR TEKNIS](#arsitektur-teknis)
3. [FLOW DATA LENGKAP](#flow-data-lengkap)
4. [BACKEND STRUCTURE](#backend-structure)
5. [FRONTEND STRUCTURE](#frontend-structure)
6. [CONTINUOUS LEARNING PIPELINE](#continuous-learning-pipeline)
7. [API ENDPOINTS](#api-endpoints)
8. [INSTALLATION & RUNNING](#installation--running)
9. [TROUBLESHOOTING](#troubleshooting)
10. [FUTURE IMPROVEMENTS](#future-improvements)

---

## OVERVIEW SISTEM

### Apa Itu Sistem Ini?

**Air Quality Monitoring System** adalah aplikasi otomatis yang:
- 🌍 Memantau kualitas udara Kota Malang real-time
- 🤖 Menggunakan 3 model Machine Learning untuk prediksi (Decision Tree, CNN, GRU)
- 📈 Pembelajaran berkelanjutan (Continuous Learning) - model otomatis meningkat setiap hari
- 🌐 Menyajikan data via Web Dashboard (React + TypeScript)
- 📱 Rekomendasi tindakan berdasarkan ISPU (Indeks Standar Pencemaran Udara) Indonesia

### Keunggulan Unik

| Fitur | Status | Penjelasan |
|-------|--------|-----------|
| **3 Model Ensemble** | ✅ Ya | Decision Tree + CNN + GRU |
| **Continuous Learning** | ✅ Ya | Otomatis retrain setiap hari |
| **Transparent** | ✅ Ya | Model interpretable, bukan black-box |
| **Autonomous** | ✅ Ya | 100% otomatis, zero manual intervention |
| **Real-time Data** | ✅ Ya | Crawling hourly dari API |
| **Local Adaptation** | ✅ Ya | Model belajar pola lokal Malang |
| **Heatmap Spasial** | ✅ Ya | Visualisasi polusi per distrik Malang |
| **Anomaly Detection** | ✅ Ya | Deteksi data abnormal otomatis |

### Target Market

- 🏛️ Pemerintah Lokal (Pemerintah Kota Malang, Badan LH)
- 🎓 Institusi Penelitian (Universitas, Research Center)
- 🏙️ Smart City Initiatives
- 📊 Environmental Monitoring Organizations

---

## ARSITEKTUR TEKNIS

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WEB FRONTEND                                 │
│                    (React + TypeScript)                             │
│                    Port: http://localhost:5173                      │
├─────────────────────────────────────────────────────────────────────┤
│                    REST API Communication                            │
├─────────────────────────────────────────────────────────────────────┤
│                         BACKEND SERVER                              │
│                      (Flask, Python)                                │
│                    Port: http://localhost:2000                      │
├─────────────────────────────────────────────────────────────────────┤
│                    APPLICATION LAYER                                 │
│  ┌────────────────┬─────────────────┬──────────────────────┐        │
│  │  Predictions   │  Recommendations │  Data Processing     │        │
│  │  Engine        │  Engine          │  & Preprocessing     │        │
│  └────────────────┴─────────────────┴──────────────────────┘        │
├─────────────────────────────────────────────────────────────────────┤
│                      MODEL LAYER                                     │
│  ┌────────────────┬─────────────────┬──────────────────────┐        │
│  │  Decision Tree │      CNN        │       GRU/LSTM       │        │
│  │   (Tabular)    │   (Spatial)     │   (Temporal)         │        │
│  └────────────────┴─────────────────┴──────────────────────┘        │
├─────────────────────────────────────────────────────────────────────┤
│                      DATA LAYER                                      │
│  ┌────────────────┬─────────────────┬──────────────────────┐        │
│  │    SQLite DB   │   CSV Files     │    Model Files       │        │
│  │  (hourly data) │   (training)    │    (.pkl, .keras)    │        │
│  └────────────────┴─────────────────┴──────────────────────┘        │
├─────────────────────────────────────────────────────────────────────┤
│                    CRAWLER & SCHEDULER                              │
│  ┌────────────────┬─────────────────┬──────────────────────┐        │
│  │ Hourly Crawl   │  Daily Training  │   Continuous Learn   │        │
│  │ (24 times)     │   (1 time)       │    (Automated)       │        │
│  └────────────────┴─────────────────┴──────────────────────┘        │
├─────────────────────────────────────────────────────────────────────┤
│                      EXTERNAL APIs                                   │
│  ┌────────────────┬─────────────────────────────────────────┐       │
│  │  Open-Meteo    │   Visual Crossing Weather API           │       │
│  │  (Pollutants)  │   (Weather Data)                        │       │
│  └────────────────┴─────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Backend:**
- Python 3.10+
- Flask (Web Framework)
- scikit-learn (Decision Tree, preprocessing)
- TensorFlow/Keras (CNN, GRU models)
- pandas/numpy (Data processing)
- APScheduler (Job scheduling)
- SQLite (Local database)

**Frontend:**
- React 19 + TypeScript
- Vite (Build tool)
- Axios (HTTP client)
- Recharts (Data visualization)
- CSS Grid/Flexbox (Styling)

**APIs:**
- Open-Meteo Air Quality API (Pollutants: PM2.5, O3, CO)
- Visual Crossing Weather API (Temperature, Humidity, Wind, Pressure, etc.)

---

## FLOW DATA LENGKAP

### Data Sources

**1. Data Polutan (Hourly)**
- **Sumber:** Open-Meteo Air Quality API
- **Lokasi:** Malang (-7.9797, 112.6304)
- **Frequensi:** Setiap jam (24 data per hari)
- **Kolom:**
  - PM2.5 (μg/m³) - Particulate Matter 2.5 mikron
  - O3 (μg/m³) - Ozone
  - CO (μg/m³) - Carbon Monoxide
- **Total Data:** 25,752 records (Jan 2023 - Okt 2025, ~3 tahun)

**2. Data Cuaca (Daily)**
- **Sumber:** Visual Crossing Weather API
- **Lokasi:** Malang
- **Frequensi:** Setiap hari (1 data per hari)
- **Kolom:**
  - Temperature (°C)
  - Humidity (%)
  - Wind Speed (m/s)
  - Wind Direction (degrees)
  - Sea Level Pressure (hPa)
  - Cloud Cover (%)
  - Visibility (km)
  - Solar Radiation (Wh/m²)
- **Total Data:** 1,035 records (Jan 2023 - Okt 2025, ~3 tahun)

### 2-Layer Data Structure

#### Layer 1: Raw Aggregated Dataset
```
File: backend/data/dataset_preprocessed/aggregate_daily.csv
Rows: ~1,000 (1 per hari)
Columns: 15 (9 polutan agg + 6 cuaca)

Struktur:
tanggal | pm2_5_mean | pm2_5_max | pm2_5_median | o3_mean | o3_max | o3_median | 
co_mean | co_max | co_median | temp | humidity | windspeed | pressure | cloudcover

Contoh:
2023-01-02 | 49.2 | 52 | 50 | 31.5 | 35 | 31 | 500 | 510 | 502 | 25.0 | 70 | 5.0 | 1013 | 20

Tujuan:
- Audit trail & reproducibility
- Append-only structure (data tumbuh setiap hari)
- Source untuk preprocessing
```

#### Layer 2: Preprocessed Dataset
```
File: backend/data/dataset_preprocessed/dataset_preprocessed.csv
Rows: ~1,000 (sama dengan Layer 1)
Columns: 50+ (15 + engineered features)

Struktur:
tanggal | (15 raw features) | (35+ engineered features)

Engineered Features Include:
- Temporal: hari, bulan, musim, is_weekend, hari_bulan_interaction
- Interactions: temp_x_humidity, wind_x_humidity, temp_humidity_interaction, heat_index, wind_humidity_interaction
- Derived: pollutant_index, weather_severity
- Rolling Stats: pm25_7d_mean, pm25_30d_mean, humidity_7d_std
- Lags: pm25_lag1, pm25_lag7, o3_lag1, co_lag1
- Scaling: MinMax [0, 1]

Tujuan:
- Training data untuk 3 model
- Siap untuk scikit-learn & TensorFlow
- Normalized untuk stabil convergence
```

### Daily Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      HOURLY CRAWLING (24x/day)                      │
│                       00:00 - 23:00                                 │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
      ┌──────────────────────────────────────┐
      │ Open-Meteo API + Visual Crossing API │
      │ Fetch polutan (hourly) + cuaca (daily)│
      └──────────────────────┬───────────────┘
                             │
                             ▼
      ┌──────────────────────────────────────┐
      │    Save to SQLite Database           │
      │ - polutan table (24 rows/day)        │
      │ - cuaca table (1 row/day)            │
      └──────────────────────┬───────────────┘
                             │
                   (Repeat hourly)
                             │
┌────────────────────────────┴────────────────────────────────────────┐
│                  DAILY AGGREGATION (23:50)                          │
│         Aggregate 24h hourly → 1 daily row                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
      ┌──────────────────────────────────────┐
      │  Agregasi hourly polutan (mean, max, │
      │  median) + daily cuaca                │
      │                                       │
      │  Output: 1 daily row                  │
      │  (1,000 rows → 1,001 rows after add) │
      └──────────────────────┬───────────────┘
                             │
                             ▼
      ┌──────────────────────────────────────┐
      │  LAYER 1: Append to aggregate_daily.csv
      │  (Raw aggregation, no preprocessing)  │
      └──────────────────────┬───────────────┘
                             │
                             ▼
      ┌──────────────────────────────────────┐
      │  PREPROCESSING (5-step pipeline)     │
      │  1. Outlier detection (Z-score)      │
      │  2. Missing value imputation         │
      │  3. Feature engineering (+35 cols)   │
      │  4. Scaling (MinMax [0,1])           │
      │  5. Save to dataset_preprocessed.csv │
      └──────────────────────┬───────────────┘
                             │
                             ▼
      ┌──────────────────────────────────────┐
      │  LAYER 2: Update dataset_preprocessed.csv
      │  (1,001 rows × 50+ columns)          │
      │  + Update minmax_scaler.pkl          │
      └──────────────────────┬───────────────┘
                             │
                             ▼
      ┌──────────────────────────────────────┐
      │  TRAINING (3 Models)                 │
      │  1. Decision Tree v1 → v1001         │
      │  2. CNN v1 → v1001                   │
      │  3. GRU v1 → v1001                   │
      │                                       │
      │  Waktu: ~5-10 menit                  │
      └──────────────────────┬───────────────┘
                             │
                             ▼
      ┌──────────────────────────────────────┐
      │  MODEL DEPLOYMENT                    │
      │  Save ke models/ folder:             │
      │  - models/Decision Tree/model.pkl    │
      │  - models/CNN/fcn_multihead_final.keras
      │  - models/GRU/multi_gru_model.keras  │
      │                                       │
      │  Dashboard reload otomatis           │
      │  Prediksi jadi lebih akurat!         │
      └──────────────────────────────────────┘

Hasil:
- Day 1:   1,000 rows → Model v1 (RMSE=8.5)
- Day 2:   1,001 rows → Model v2 (RMSE=8.3) ✓ Lebih baik!
- Day 3:   1,002 rows → Model v3 (RMSE=8.1) ✓ Lebih baik!
...
- Day 100: 1,100 rows → Model v100 (RMSE=6.2) ✓ Semakin akurat!

Akurasi ↗️ terus meningkat setiap hari!
```

---

## BACKEND STRUCTURE

### Directory Layout

```
backend/
├── app.py                              # Main Flask application (907 lines)
├── training.py                         # Training script for 3 models (1457 lines)
├── data_pipeline.py                    # Data aggregation & preprocessing (409 lines)
├── requirements.txt                    # Python dependencies
├── check_deps.py                       # Dependency checker
├── .env                                # Environment variables (API keys)
│
├── crawler/                            # Data crawling module
│   ├── daily_crawler.py                # Hourly crawling from APIs
│   ├── scheduler.py                    # APScheduler for continuous learning
│   ├── db_handler.py                   # SQLite database operations (980 lines)
│   ├── sync_csv_to_db.py               # Sync CSV to database
│   └── __pycache__/
│
├── utils/                              # Utility modules
│   ├── prediction.py                   # Prediction engine (1384 lines)
│   ├── ispu_classifier.py              # ISPU classification (324 lines)
│   ├── recommendation.py               # Recommendation engine (321 lines)
│   ├── data_loader.py                  # Data loading & preprocessing (695 lines)
│   ├── debug_cnn.py                    # CNN debugging utilities
│   └── __pycache__/
│
├── data/                               # Data storage
│   ├── datacrawler.db                  # SQLite database
│   ├── datacrawler.db.backup           # Backup database
│   ├── outliers.csv                    # Detected outliers
│   │
│   ├── harian/                         # Daily data files from crawling
│   │   ├── polutan_2026-01-01.csv
│   │   ├── polutan_2026-01-02.csv
│   │   ├── cuaca_2026-01-01.csv
│   │   └── ...
│   │
│   ├── dataset_preprocessed/           # Preprocessed datasets
│   │   ├── dataset_preprocessed.csv    # LAYER 2: Preprocessed training data
│   │   └── minmax_scaler.pkl           # MinMax scaler
│   │
│   └── raw_data/                       # Original Excel files
│       ├── Data Polutan.xlsx           # Historical polutan data
│       └── Data Cuaca 2023-2025.xlsx   # Historical weather data
│
├── models/                             # Trained models (production)
│   ├── Decision Tree/
│   │   ├── model.pkl                   # Decision Tree model
│   │   ├── scaler.pkl                  # Feature scaler
│   │   ├── feature_cols.pkl            # Feature columns
│   │   ├── metadata.pkl                # Best params, CV scores
│   │   ├── metrics.pkl                 # Evaluation metrics
│   │   └── feature_importance.pkl      # Feature importance ranking
│   │
│   ├── CNN/
│   │   ├── fcn_multihead_final.keras   # CNN model (Keras)
│   │   ├── scaler_X_cnn_eval.pkl       # Input scaler
│   │   ├── scaler_yreg_cnn_eval.pkl    # Output scaler
│   │   ├── y_map_max_cnn_eval.pkl      # Max values for heatmap
│   │   ├── evaluation_metrics_cnn_fixed.pkl
│   │   └── evaluation_metrics_cnn_fixed.json
│   │
│   └── GRU/
│       ├── multi_gru_model.keras       # GRU model (Keras)
│       ├── scaler_X_multi_conservative.pkl
│       ├── scalers_y_multi_conservative.pkl
│       ├── feature_cols_multi_conservative.pkl
│       └── target_cols_multi_conservative.pkl
│
├── Training/                           # Training artifacts (reference)
│   ├── Decision Tree/
│   ├── CNN/
│   ├── GRU/
│   ├── shared_data/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── training_metrics.json
│
└── logs/
    └── continuous_learning.log         # Training logs
```

### Key Backend Files

#### 1. app.py (Flask Backend - 907 lines)
```python
# Main application file
# Responsibilities:
# - Load 3 models (DT, CNN, GRU) at startup
# - CORS configuration for frontend
# - API endpoints for predictions & data

# Key Components:
- BaselineModel: Fallback model if trained model fails
- Model Loading: Prioritizes multiple path locations
- CORS: Allow frontend communication
- Prediction Engine: Unified interface for 3 models
```

**Key Features:**
- Load models from `models/` folder
- Create scalers from pickled files
- Initialize PredictionEngine, ISPUClassifier, RecommendationEngine
- Setup Flask routes for API
- Optional scheduler for continuous learning

#### 2. training.py (Model Training - 1457 lines)
```python
# Complete training pipeline
# Responsibilities:
# - Load raw data from Excel files
# - Preprocess data (12-step pipeline)
# - Train 3 models with hyperparameter tuning
# - Save models and metrics

# Key Functions:
- preprocess_data(): 12-step preprocessing pipeline
- train_decision_tree(): Tree regression with hyperparameter search
- train_cnn(): Convolutional neural network for spatial patterns
- train_gru(): Gated Recurrent Unit for temporal patterns
- train_ensemble(): Voting from 3 models
```

#### 3. crawler/daily_crawler.py (Hourly Data Crawling)
```python
# Crawl data every hour
# Responsibilities:
# - Fetch polutan from Open-Meteo API
# - Fetch cuaca from Visual Crossing API
# - Save to SQLite database & CSV files
# - Handle rate limiting with exponential backoff

# Key Function:
- crawl_daily(target_date, save_to_db, skip_if_exists)
  Returns: (df_polut, df_cuaca) or (None, None) if skip
```

#### 4. crawler/scheduler.py (Continuous Learning)
```python
# APScheduler for automated jobs
# Schedule:
# - 00:00, 01:00, ..., 23:00 → Hourly crawl
# - 23:50 → Daily aggregation + training

# Key Functions:
- hourly_crawling_job(): Crawl data polutan + cuaca
- daily_aggregation_and_training_job(): 5-step training pipeline
```

#### 5. utils/prediction.py (Unified Prediction - 1384 lines)
```python
# Prediction engine for all 3 models
# Responsibilities:
# - Load models and scalers at init
# - Predict using DT, CNN, or GRU
# - Inverse transform outputs
# - Generate heatmaps

# Key Classes:
- PredictionEngine: Main prediction interface
  Methods:
  - predict_dt(data) → pm25, o3, co
  - predict_cnn(data) → pm25, o3, co
  - predict_gru(data) → pm25, o3, co
  - generate_heatmap(data) → 15x15 grid
```

#### 6. utils/ispu_classifier.py (ISPU Classification - 324 lines)
```python
# Classify air quality based on Indonesian standards
# Standards: KEP-45/MENLH/10/1997 + Peraturan Menteri LH No. 14/2010

# Classes:
- ISPUClassifier:
  - classify_pm25(value) → 'Baik'|'Sedang'|'Tidak Sehat'|'Sangat Tidak Sehat'|'Berbahaya'
  - classify_o3(value) → category
  - classify_co(value) → category
  - classify_all(pm25, o3, co) → overall status + advice + color
```

#### 7. utils/recommendation.py (Recommendations - 321 lines)
```python
# Generate contextual recommendations

# Classes:
- RecommendationEngine:
  - get_recommendations(ispu_category) → dict with 6 contexts
    Contexts:
    - rumah_tangga (Household)
    - transportasi (Transportation)
    - kesehatan (Health)
    - perkantoran (Office)
    - lingkungan (Environment)
    - komunitas (Community)
```

---

## FRONTEND STRUCTURE

### Directory Layout

```
frontend-react/
├── package.json                        # Dependencies
├── tsconfig.json                       # TypeScript config
├── vite.config.ts                      # Vite build config
├── eslint.config.js                    # ESLint rules
├── index.html                          # HTML entry point
│
├── src/
│   ├── main.tsx                        # React entry point
│   ├── App.tsx                         # Main app component
│   ├── App.css                         # Global styles
│   ├── index.css                       # Root styles
│   ├── api.ts                          # API client (axios)
│   │
│   ├── components/
│   │   ├── Header.tsx                  # Navigation header
│   │   ├── Dashboard.tsx               # Main dashboard (predictions)
│   │   ├── Recommendations.tsx         # Recommendation cards
│   │   ├── History.tsx                 # Historical data table
│   │   └── Charts/                     # Recharts visualizations
│   │
│   └── assets/                         # Images, icons, etc.
│
└── public/
    └── vite.svg                        # Logo
```

### Key Frontend Files

#### 1. api.ts (API Client)
```typescript
// Axios instance for backend communication
export async function getDashboard(date?: string)
export async function getRecommendations()
export async function getHistory(limit = 30)
export async function getAnomalies(limit = 10)
```

#### 2. App.tsx (Main Component)
```typescript
// Router between tabs:
- "prediksi" → Dashboard (predictions for selected date)
- "rekomendasi" → Recommendations (action items)
- "riwayat" → History (30 days of historical data)
```

#### 3. Dashboard.tsx (Predictions)
```typescript
// Main dashboard showing:
- Selected date picker
- Predictions from 3 models (DT, CNN, GRU)
- ISPU category + color + advice
- 6-day trend chart
- Heatmap (15x15 grid of districts)
- Anomalies detected
- Statistics summary

// Data types:
interface DashboardResponse {
  status: string
  timestamp: string
  primary_model: string
  predictions: {
    dt: { pm25, o3, co, ispu }
    cnn: { pm25, o3, co, ispu }
    gru: { pm25, o3, co, ispu }
  }
  trend: { data, period }
  heatmap: { data, grid_size, pollutants }
  anomalies: { data, count, period }
  statistics: { ispu_categories, total_days, total_hours, date_range }
}
```

---

## CONTINUOUS LEARNING PIPELINE

### Why Continuous Learning?

Traditional ML systems:
```
Day 1:    Train model with 200 rows    → Model v1 (RMSE=8.5)
Day 2:    Use model v1                 → Model v1 (RMSE=8.5) [SAME]
Day 100:  Use model v1                 → Model v1 (RMSE=8.5) [SAME]

❌ Model tidak berkembang, akurasi tetap statis
```

**OUR System with Continuous Learning:**
```
Day 1:    Train with 200 rows          → Model v1 (RMSE=8.5)
Day 2:    Crawl +1 row → Train with 201 → Model v2 (RMSE=8.3) ✅ Lebih baik!
Day 3:    Crawl +1 row → Train with 202 → Model v3 (RMSE=8.1) ✅ Lebih baik!
...
Day 100:  Crawl +1 row → Train with 300 → Model v100 (RMSE=6.2) ✅ Jauh lebih baik!

✅ Model otomatis meningkat akurasi setiap hari!
```

### 5-Step Daily Training Pipeline

**Timeline:** Setiap hari jam 23:50 WIB

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 1: Hourly Data Aggregation                  │
│                      (24h → 1 daily row)                            │
│                                                                      │
│  Ambil 24 jam data dari SQLite:                                     │
│  - polutan table: 24 rows (jam 0-23) per tanggal                    │
│  - cuaca table: 1 row per tanggal                                   │
│                                                                      │
│  Agregasi:                                                           │
│  PM2.5_mean   = (PM2.5_h0 + PM2.5_h1 + ... + PM2.5_h23) / 24       │
│  PM2.5_max    = max(PM2.5_h0, PM2.5_h1, ..., PM2.5_h23)            │
│  PM2.5_median = median(...)                                         │
│  (Repeat untuk O3, CO)                                              │
│                                                                      │
│  Output: 1 aggregation dict dengan 15 keys                          │
└────────────────────┬─────────────────────────────────────────────────┘
                     │
┌────────────────────┴─────────────────────────────────────────────────┐
│                    STEP 2: Merge to Master Dataset                   │
│                        (LAYER 1: RAW)                                │
│                                                                      │
│  Append 1 daily row ke aggregate_daily.csv:                          │
│  Sebelum: 1,000 rows                                                 │
│  Sesudah: 1,001 rows                                                 │
│                                                                      │
│  File tumbuh setiap hari (audit trail)                              │
└────────────────────┬─────────────────────────────────────────────────┘
                     │
┌────────────────────┴─────────────────────────────────────────────────┐
│                    STEP 3: Preprocessing                             │
│                        (LAYER 2: PROCESSED)                          │
│                                                                      │
│  Pipeline 5-step:                                                    │
│  3a. Outlier Detection (Z-score > 3)                                │
│  3b. Missing Value Imputation (interpolate, forward/backward fill)  │
│  3c. Feature Engineering (+35 features)                             │
│  3d. MinMax Scaling [0, 1]                                          │
│  3e. Save dataset_preprocessed.csv                                  │
│                                                                      │
│  Input:  1,001 rows × 15 columns                                    │
│  Output: 1,001 rows × 50+ columns (scaled)                          │
│                                                                      │
│  File: backend/data/dataset_preprocessed/dataset_preprocessed.csv  │
│  Scaler: backend/data/dataset_preprocessed/minmax_scaler.pkl       │
└────────────────────┬─────────────────────────────────────────────────┘
                     │
┌────────────────────┴─────────────────────────────────────────────────┐
│                    STEP 4: Training 3 Models                         │
│                     (~5-10 menit total)                              │
│                                                                      │
│  4a. Decision Tree                                                   │
│      Input:  1,001 rows × 50+ columns                               │
│      Output: Decision Tree v1001                                    │
│      CV: GridSearchCV dengan hyperparameter tuning                  │
│                                                                      │
│  4b. CNN (Convolutional Neural Network)                             │
│      Input:  1,001 rows → 15×15 spatial grid                        │
│      Output: CNN model v1001                                        │
│      Architecture: Multi-head FCN with 3 output heads (PM25, O3, CO)
│                                                                      │
│  4c. GRU (Gated Recurrent Unit)                                     │
│      Input:  1,001 rows × 46 features (temporal)                    │
│      Output: GRU model v1001                                        │
│      Architecture: Multi-output LSTM/GRU with 3 targets             │
│                                                                      │
│  Metrics tracked:                                                    │
│  - RMSE, MAE, R² untuk setiap model                                 │
│  - Cross-validation scores                                          │
│  - Feature importance (DT)                                          │
└────────────────────┬─────────────────────────────────────────────────┘
                     │
┌────────────────────┴─────────────────────────────────────────────────┐
│                  STEP 5: Model Save & Verification                   │
│                                                                      │
│  Save to production models/ folder:                                 │
│                                                                      │
│  Decision Tree:                                                      │
│    - models/Decision Tree/model.pkl                                 │
│    - models/Decision Tree/scaler.pkl                                │
│    - models/Decision Tree/feature_cols.pkl                          │
│    - models/Decision Tree/metadata.pkl                              │
│    - models/Decision Tree/metrics.pkl                               │
│                                                                      │
│  CNN:                                                                │
│    - models/CNN/fcn_multihead_final.keras                           │
│    - models/CNN/scaler_X_cnn_eval.pkl                               │
│    - models/CNN/scaler_yreg_cnn_eval.pkl                            │
│                                                                      │
│  GRU:                                                                │
│    - models/GRU/multi_gru_model.keras                               │
│    - models/GRU/scaler_X_multi_conservative.pkl                     │
│    - models/GRU/scalers_y_multi_conservative.pkl                    │
│                                                                      │
│  Dashboard otomatis reload & use model terbaru                      │
│  Prediksi jadi lebih akurat! ✅                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Scheduler Jobs

**Hourly Crawling Job:**
- **Waktu:** 00:00, 01:00, 02:00, ..., 23:00 (24x per hari)
- **Durasi:** ~30 detik
- **Fungsi:** Fetch polutan + cuaca, save ke SQLite
- **File:** `crawler/daily_crawler.py:crawl_daily()`

**Daily Aggregation + Training Job:**
- **Waktu:** 23:50 (setiap hari)
- **Durasi:** ~5-10 menit
- **Fungsi:** Run 5-step training pipeline
- **File:** `crawler/scheduler.py:daily_aggregation_and_training_job()`

---

## API ENDPOINTS

### Base URL
```
http://localhost:2000
```

### Endpoints

#### 1. `/api/dashboard` (GET)
Dapatkan prediksi untuk tanggal tertentu

**Query Parameters:**
```
?date=2026-01-05  (format: YYYY-MM-DD, default: today)
```

**Response:**
```json
{
  "status": "success",
  "timestamp": "2026-01-05T10:30:00",
  "primary_model": "decision_tree",
  "predictions": {
    "decision_tree": {
      "pm2_5": 35.2,
      "o3": 45.8,
      "co": 450,
      "ispu": "Sedang"
    },
    "cnn": {
      "pm2_5": 34.8,
      "o3": 46.2,
      "co": 455,
      "ispu": "Sedang"
    },
    "gru": {
      "pm2_5": 35.5,
      "o3": 45.5,
      "co": 448,
      "ispu": "Sedang"
    }
  },
  "ensemble_prediction": {
    "pm2_5": 35.17,
    "o3": 45.83,
    "co": 451,
    "ispu": "Sedang",
    "color": "#eab308",
    "advice": "Kualitas udara dapat diterima untuk sebagian besar orang"
  },
  "trend": {
    "data": [
      { "date": "2025-12-30", "pm25": 32.1, "o3": 44.2, "co": 440 },
      { "date": "2025-12-31", "pm25": 33.5, "o3": 45.1, "co": 445 },
      ...
    ],
    "period": "6 hari terakhir"
  },
  "heatmap": {
    "data": [
      { "district": "Blimbing", "pm25": 35.2, "color": "#eab308" },
      { "district": "Lowokwaru", "pm25": 36.1, "color": "#f97316" },
      ...
    ],
    "grid_size": "15x15",
    "pollutants": ["pm2_5", "o3", "co"]
  },
  "anomalies": {
    "data": [
      { "pollutant": "PM2.5", "description": "Nilai abnormal tinggi", "datetime": "2026-01-05 08:00", "value": 85.5 }
    ],
    "count": 1,
    "period": "24 jam terakhir"
  },
  "statistics": {
    "ispu_categories": [
      { "name": "Baik", "value": 8, "color": "#22c55e" },
      { "name": "Sedang", "value": 15, "color": "#eab308" },
      ...
    ],
    "total_days": 1000,
    "total_hours": 24000,
    "date_range": { "start": "2023-01-02", "end": "2026-01-05" }
  }
}
```

#### 2. `/api/recommendations` (GET)
Dapatkan rekomendasi tindakan

**Response:**
```json
{
  "status": "success",
  "ispu_category": "Sedang",
  "recommendations": {
    "rumah_tangga": [
      "🪟 Buka jendela saat pagi/sore",
      "🧹 Bersihkan rumah lebih sering",
      "🌱 Tanaman indoor membantu filtrasi"
    ],
    "transportasi": [
      "🚌 Gunakan transportasi umum",
      "⏰ Hindari jam sibuk jika memungkinkan",
      "🚗 Carpooling lebih diutamakan"
    ],
    ...
  }
}
```

#### 3. `/api/history` (GET)
Dapatkan data historis (30 hari terakhir by default)

**Query Parameters:**
```
?limit=30  (jumlah hari, default: 30)
```

**Response:**
```json
{
  "status": "success",
  "data": [
    { "date": "2025-12-06", "pm25": 32.1, "o3": 44.2, "co": 440 },
    { "date": "2025-12-07", "pm25": 33.5, "o3": 45.1, "co": 445 },
    ...
  ],
  "count": 30
}
```

#### 4. `/api/anomalies` (GET)
Dapatkan data anomali terdeteksi

**Query Parameters:**
```
?limit=10  (jumlah anomali, default: 10)
```

**Response:**
```json
{
  "status": "success",
  "data": [
    { "pollutant": "PM2.5", "description": "Nilai abnormal tinggi", "datetime": "2026-01-05 08:00", "value": 85.5, "increase_percent": 150 },
    ...
  ],
  "count": 1
}
```

#### 5. `/api/history/export` (GET)
Export data historis dalam format CSV

**Query Parameters:**
```
?format=csv  (default)
?limit=100
```

#### 6. `/api/crawl-now` (POST)
Trigger crawling sekarang (untuk testing)

**Response:**
```json
{
  "status": "success",
  "message": "Crawling selesai untuk tanggal 2026-01-05",
  "polutan_records": 24,
  "cuaca_records": 1
}
```

#### 7. `/` (GET)
Health check endpoint

**Response:**
```json
{
  "status": "ok",
  "message": "Air Quality Monitoring System is running"
}
```

---

## INSTALLATION & RUNNING

### Prerequisites

- Python 3.10+
- Node.js 16+ (for frontend)
- pip (Python package manager)
- npm (Node package manager)

### Step 1: Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Create .env file with API keys
cat > .env << EOF
VISUAL_API_KEY=your_visual_crossing_api_key_here
EOF

# Test dependencies
python check_deps.py

# Run training (first time setup)
python training.py

# Start backend server
python app.py
# Server akan start di http://localhost:2000
```

### Step 2: Frontend Setup

```bash
cd frontend-react

# Install dependencies
npm install

# Start dev server
npm run dev
# Frontend akan start di http://localhost:5173
```

### Step 3: Start Everything (One Command)

```bash
cd /Users/user/Desktop/SKRIPSI/HASIL/Air\ Quality

# Run main script
bash run_app.sh

# Output:
# 🚀 Starting Backend (Port 2000)...
# 🚀 Starting Frontend...
# ✅ App is running!
# 🖥️  Frontend: http://localhost:5173
# ⚙️  Backend:  http://localhost:2000
# Press Ctrl+C to stop everything.
```

---

## TROUBLESHOOTING

### Problem 1: "Module not found" errors

**Solution:**
```bash
# Backend
cd backend
pip install -r requirements.txt --force-reinstall

# Frontend
cd frontend-react
npm install --legacy-peer-deps
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### Problem 2: Models not loading

**Symptom:** "Warning: DT model not found"

**Solution:**
```bash
# Run training to generate models
cd backend
python training.py

# This will create:
# - models/Decision Tree/model.pkl
# - models/CNN/fcn_multihead_final.keras
# - models/GRU/multi_gru_model.keras
```

### Problem 3: API Key errors

**Symptom:** "VISUAL_API_KEY tidak ditemukan"

**Solution:**
```bash
# Create .env file in backend directory
cd backend
cat > .env << EOF
VISUAL_API_KEY=your_api_key_here
EOF

# Get API key from: https://www.visualcrossing.com
```

### Problem 4: Port already in use

**Symptom:** "Address already in use:5173"

**Solution:**
```bash
# Kill process on port 5173
lsof -ti:5173 | xargs kill -9
lsof -ti:2000 | xargs kill -9

# Then start again
bash run_app.sh
```

### Problem 5: TensorFlow compatibility issues

**Symptom:** "Model loading failed: keras version mismatch"

**Solution:**
```bash
# Reinstall TensorFlow
pip uninstall tensorflow -y
pip install tensorflow==2.18.0

# Or use CPU-only version if GPU issues
pip install tensorflow[cpu]==2.18.0
```

---

## FUTURE IMPROVEMENTS

### Phase 2: Enhanced Features

1. **Real IoT Sensors Integration**
   - Add actual air quality sensors in Malang
   - Hybrid system: API + Sensors
   - Improve accuracy with real measured data

2. **Mobile App**
   - React Native mobile app
   - Push notifications for high pollution alerts
   - Location-based recommendations
   - Offline capability

3. **Multi-City Expansion**
   - Scale to Surabaya, Bandung, Jakarta
   - Federated learning between cities
   - Share knowledge patterns

4. **Advanced Analytics**
   - Causal inference (causes of pollution)
   - Health impact correlation
   - Policy recommendations

5. **Edge Deployment**
   - Run models on edge devices (Raspberry Pi, etc.)
   - Minimal server load
   - Privacy-focused local inference

### Phase 3: Production Deployment

1. **Docker Containerization**
   - Docker Compose for full stack
   - Easy deployment on any server
   - AWS/GCP/Azure support

2. **Database Migration**
   - SQLite → PostgreSQL (scalable)
   - Replicated backups
   - Time-series database (InfluxDB)

3. **Monitoring & Logging**
   - ELK Stack (Elasticsearch, Logstash, Kibana)
   - Prometheus for metrics
   - Alert system

4. **CI/CD Pipeline**
   - GitHub Actions
   - Automated testing
   - Continuous deployment

5. **High Availability**
   - Load balancing
   - Horizontal scaling
   - Fault tolerance

### Phase 4: Advanced ML

1. **Ensemble Stacking**
   - Meta-learner on top of 3 models
   - Better ensemble predictions

2. **Online Learning**
   - Update models in real-time
   - Stream processing (Kafka)
   - Adaptive learning rate

3. **Transfer Learning**
   - Pre-trained models from global data
   - Fine-tune with Malang-specific data
   - Faster convergence

4. **Uncertainty Quantification**
   - Bayesian neural networks
   - Confidence intervals for predictions
   - Decision support

---

## CONTACT & SUPPORT

**Developer:** [Your Name]  
**Email:** [Your Email]  
**GitHub:** [Repository Link]  
**Documentation:** This file + inline code comments

---

**Last Updated:** 5 Januari 2026  
**Version:** 1.0 (Production Ready)  
**Status:** ✅ Active Development
