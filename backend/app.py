from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import sys
import importlib
import logging

# Setup logging untuk continuous learning
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/continuous_learning.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Compatibility shim: some installed libraries expect
# importlib.metadata.packages_distributions (added in newer Python).
# If it's missing on this Python version, try to provide it from the
# importlib_metadata backport so dependent libs don't raise AttributeError.
try:
    import importlib.metadata as _std_meta
    if not hasattr(_std_meta, 'packages_distributions'):
        try:
            import importlib_metadata as _backport_meta
            setattr(_std_meta, 'packages_distributions', _backport_meta.packages_distributions)
            # also expose in importlib.metadata name
            importlib.metadata = _std_meta
            print("ℹ️ Patched importlib.metadata.packages_distributions from backport")
        except Exception:
            # best-effort; continue without shim
            print("⚠️ importlib.metadata.packages_distributions not available and backport not found")
except Exception:
    # ignore if importlib.metadata isn't available
    pass

# Import scheduler untuk continuous learning
try:
    from crawler.scheduler import start_scheduler
    SCHEDULER_AVAILABLE = True
    logger.info("[APP] ✅ Scheduler module imported successfully")
except Exception as e:
    SCHEDULER_AVAILABLE = False
    logger.warning(f"[APP] ⚠️ Scheduler module tidak tersedia: {e}")

# ============================================================================
# IMPORT MODELING DEPENDENCIES
# ============================================================================
try:
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available, CNN/LSTM features disabled")
    KERAS_AVAILABLE = False

# Custom utilities
try:
    from utils.prediction import PredictionEngine
    from utils.ispu_classifier import ISPUClassifier
    from utils.recommendation import RecommendationEngine
    from utils.data_loader import DataLoader
    from crawler.db_handler import get_trend_data_from_db, get_ispu_statistics_from_db, get_anomalies_from_db, get_history_data
except ImportError as e:
    print(f"❌ Error importing utilities: {e}")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================
app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://localhost:5173", "*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================================
# LOAD MODELS & DATA
# ============================================================================
print("="*80)
print("🚀 LOADING MODELS & DATA...")
print("="*80)

dt_model = cnn_model = lstm_model = None
cnn_scaler = lstm_scaler = scaler = None
train_df = val_df = test_df = outliers_df = None


class BaselineModel:
    """Very small fallback model with a predict method.

    If a saved model cannot be loaded due to binary incompatibilities, we
    use this baseline which returns either the incoming pm2.5 feature (if
    present) or a constant baseline value.
    """
    def __init__(self, baseline_value: float = 25.0):
        self.baseline_value = float(baseline_value)

    def predict(self, X):
        # X expected shape (n_samples, n_features)
        try:
            arr = np.array(X)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] >= 1:
                # assume first column corresponds to pm2.5 original feature
                # (may be scaled) — return it as-is, otherwise fallback
                return arr[:, 0]
        except Exception:
            pass
        return np.full((arr.shape[0] if 'arr' in locals() else 1,), self.baseline_value)

# --- Decision Tree
try:
    # Prioritas 1: models/Decision Tree/ (struktur baru)
    dt_path = os.path.join(MODEL_DIR, 'Decision Tree', 'model.pkl')
    if not os.path.exists(dt_path):
        # Prioritas 2: models/ (legacy)
        dt_path = os.path.join(MODEL_DIR, 'dt_balanced_model.pkl')
    if not os.path.exists(dt_path):
        dt_path = os.path.join(MODEL_DIR, 'decision_tree_model.pkl')
    
    if os.path.exists(dt_path):
        dt_model = joblib.load(dt_path)
        print(f"✅ Decision Tree loaded: {os.path.basename(dt_path)}")
        
        # Load scaler dari lokasi yang sama
        scaler_path = os.path.join(os.path.dirname(dt_path), 'scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
                print(f"✅ Decision Tree scaler loaded: {scaler_path}")
            except Exception as e:
                print(f"⚠️ DT scaler load error: {e}")
    else:
        # Fallback: try Training folder
        training_dt = os.path.join(BASE_DIR, 'Training', 'Decision Tree', 'model.pkl')
        if os.path.exists(training_dt):
            try:
                dt_model = joblib.load(training_dt)
                print(f"✅ Decision Tree loaded from Training: {training_dt}")
                
                # Also try to load DT scaler
                training_dt_scaler = os.path.join(BASE_DIR, 'Training', 'Decision Tree', 'scaler.pkl')
                if os.path.exists(training_dt_scaler):
                    try:
                        scaler = joblib.load(training_dt_scaler)
                        print(f"✅ Decision Tree scaler loaded from Training: {training_dt_scaler}")
                    except Exception as e:
                        print(f"⚠️ DT scaler load error from Training: {e}")
            except Exception as e:
                print(f"❌ DT load error from Training: {e}")
except Exception as e:
    print(f"❌ DT load error: {e}")

# --- CNN
if KERAS_AVAILABLE:
    try:
        # Prioritas 1: models/CNN/ (struktur baru)
        cnn_path = os.path.join(MODEL_DIR, 'CNN', 'best_cnn.keras')
        if not os.path.exists(cnn_path):
            # Prioritas 2: models/ (legacy)
            cnn_path = os.path.join(MODEL_DIR, 'cnn_3h_model.h5')
        if not os.path.exists(cnn_path):
            cnn_path = os.path.join(MODEL_DIR, 'cnn_spatial_model.h5')
        
        if os.path.exists(cnn_path):
            try:
                # Load without custom objects - just for inference, no training
                cnn_model = keras.models.load_model(cnn_path, compile=False)
                print(f"✅ CNN model loaded: {os.path.basename(cnn_path)}")
            except Exception as e:
                print(f"⚠️ Could not load CNN from {cnn_path}: {e}")
                cnn_model = None
        else:
            # Fallback: try Training folder
            training_cnn_candidates = [
                os.path.join(BASE_DIR, 'Training', 'CNN', 'best_cnn.keras'),
                os.path.join(BASE_DIR, 'Training', 'CNN', 'best_model.pkl'),
                os.path.join(BASE_DIR, 'Training', 'CNN', 'enhanced_cnn_tuned_model.pkl')
            ]
            for cand in training_cnn_candidates:
                if os.path.exists(cand):
                    try:
                        if cand.endswith('.keras') or cand.endswith('.h5'):
                            cnn_model = keras.models.load_model(cand, compile=False)
                            print(f"✅ CNN model loaded from Training: {cand}")
                            break
                        else:
                            loaded_obj = joblib.load(cand)
                            # Verify it's actually a model, not a scaler
                            if hasattr(loaded_obj, 'predict') and not hasattr(loaded_obj, 'transform'):
                                cnn_model = loaded_obj
                                print(f"✅ CNN model loaded from Training: {cand}")
                                break
                            else:
                                print(f"⚠️ {cand} is not a CNN model (it's a {type(loaded_obj).__name__})")
                    except Exception as e:
                        # Keras 3.x incompatibility or other errors
                        print(f"⚠️ Could not load CNN from {os.path.basename(cand)}: {e}")
            
            if cnn_model is None:
                print("⚠️ CNN model not available")
        
        # Load CNN scaler separately
        # Prioritas 1: models/CNN/ (struktur baru)
        scaler_path = os.path.join(MODEL_DIR, 'CNN', 'scaler_X.pkl')
        if not os.path.exists(scaler_path):
            # Prioritas 2: models/ (legacy)
            scaler_path = os.path.join(MODEL_DIR, 'cnn_grid_scaler.pkl')
        
        if os.path.exists(scaler_path):
            try:
                cnn_scaler = joblib.load(scaler_path)
                print(f"✅ CNN scaler loaded: {scaler_path}")
            except Exception as e:
                print(f"⚠️ CNN scaler load error: {e}")
        else:
            # Fallback: try loading CNN scaler from Training
            training_cnn_scaler = os.path.join(BASE_DIR, 'Training', 'CNN', 'scaler_X.pkl')
            if not os.path.exists(training_cnn_scaler):
                training_cnn_scaler = os.path.join(BASE_DIR, 'Training', 'CNN', 'enhanced_scaler.pkl')
            if os.path.exists(training_cnn_scaler):
                try:
                    cnn_scaler = joblib.load(training_cnn_scaler)
                    print(f"✅ CNN scaler loaded from Training: {training_cnn_scaler}")
                except Exception as e:
                    print(f"⚠️ CNN scaler load error from Training: {e}")
    except Exception as e:
        print(f"⚠️ CNN load error: {e}")

# Load CNN Y-scaler (output scaler)
cnn_y_scaler = None
try:
    cnn_y_scaler_candidates = [
        os.path.join(MODEL_DIR, 'CNN', 'scaler_yreg.pkl'),
        os.path.join(MODEL_DIR, 'cnn_y_scaler.pkl'),
    ]
    for cand in cnn_y_scaler_candidates:
        if os.path.exists(cand):
            try:
                cnn_y_scaler = joblib.load(cand)
                print(f"✅ CNN y-scaler loaded: {cand}")
                break
            except Exception as e:
                print(f"⚠️ CNN y-scaler load error from {cand}: {e}")
except Exception as e:
    print(f"⚠️ CNN y-scaler load error: {e}")

# --- GRU/LSTM
if KERAS_AVAILABLE:
    try:
        # Prioritas 1: models/GRU/ (struktur baru)
        gru_path = os.path.join(MODEL_DIR, 'GRU', 'model_v9.keras')
        if not os.path.exists(gru_path):
            # Prioritas 2: models/ (legacy)
            gru_path = os.path.join(MODEL_DIR, 'lstm_model.h5')
        
        if os.path.exists(gru_path):
            try:
                lstm_model = keras.models.load_model(gru_path)
                print(f"✅ GRU/LSTM model loaded: {os.path.basename(gru_path)}")
            except Exception as e:
                print(f"⚠️ Could not load GRU from {gru_path}: {e}")
                lstm_model = None
        else:
            print("⚠️ GRU model not found in models folder")
        
        # Load GRU scaler
        # Prioritas 1: models/GRU/ (struktur baru)
        gru_scaler_path = os.path.join(MODEL_DIR, 'GRU', 'scaler_X.pkl')
        if not os.path.exists(gru_scaler_path):
            # Prioritas 2: models/ (legacy)
            gru_scaler_path = os.path.join(MODEL_DIR, 'lstm_scaler.pkl')
        
        if os.path.exists(gru_scaler_path):
            try:
                lstm_scaler = joblib.load(gru_scaler_path)
                print(f"✅ GRU scaler loaded: {gru_scaler_path}")
            except Exception as e:
                print(f"⚠️ GRU scaler load error: {e}")
        
        # Fallback: try loading GRU/LSTM from Training folder
        if lstm_model is None:
            training_gru_candidates = [
                os.path.join(BASE_DIR, 'Training', 'GRU', 'model_v9.keras'),
                os.path.join(BASE_DIR, 'Training', 'GRU', 'gru_final_model.pkl')
            ]
            for cand in training_gru_candidates:
                if os.path.exists(cand):
                    try:
                        if cand.endswith('.keras') or cand.endswith('.h5'):
                            lstm_model = keras.models.load_model(cand)
                            print(f"✅ GRU model loaded from Training: {cand}")
                            break
                        else:
                            lstm_model = joblib.load(cand)
                            print(f"✅ GRU model loaded from Training: {cand}")
                            break
                    except Exception as e:
                        if 'keras.src' in str(e):
                            print(f"⚠️ GRU model is Keras 3.x, incompatible with TensorFlow 2.15 environment")
                        else:
                            print(f"⚠️ Could not load GRU from Training: {e}")
            
            # Load scaler from Training
            if lstm_scaler is None:
                training_gru_scaler_candidates = [
                    os.path.join(BASE_DIR, 'Training', 'GRU', 'scaler_X.pkl'),
                    os.path.join(BASE_DIR, 'Training', 'GRU', 'scaler_X_gru.pkl')
                ]
                for cand in training_gru_scaler_candidates:
                    if os.path.exists(cand):
                        try:
                            lstm_scaler = joblib.load(cand)
                            print(f"✅ GRU scaler loaded from Training: {cand}")
                            break
                        except Exception as e:
                            print(f"⚠️ Could not load GRU scaler from Training: {e}")
    except Exception as e:
        print(f"⚠️ GRU/LSTM load error: {e}")

# --- Feature scaler (general)
try:
    scaler_path = os.path.join(DATA_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("✅ Feature scaler loaded")
except Exception as e:
    print(f"⚠️ Feature scaler error: {e}")

# --- Load datasets
try:
    data_loader = DataLoader(DATA_DIR)
    train_df = data_loader.load_train()
    val_df = data_loader.load_val()
    test_df = data_loader.load_test()
    outliers_df = data_loader.load_outliers()
    print(f"✅ Data loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
except Exception as e:
    print(f"❌ Data load error: {e}")
    data_loader = None

# If decision tree model wasn't loaded, provide a simple baseline model
# constructed from dataset averages (so API still returns reasonable values)
try:
    if dt_model is None and data_loader is not None:
        all_df = data_loader.get_all_data()
        baseline_pm25 = 25.0
        if not all_df.empty and 'pm2_5 (μg/m³)_mean' in all_df.columns:
            try:
                baseline_pm25 = float(all_df['pm2_5 (μg/m³)_mean'].mean())
            except Exception:
                baseline_pm25 = 25.0
        dt_model = BaselineModel(baseline_pm25)
        print(f"ℹ️ Decision Tree fallback: using BaselineModel (pm2.5 baseline={baseline_pm25:.2f})")

    # For LSTM/GRU fallback, we let PredictionEngine handle it (it has better simulated logic)
    # if lstm_model is None and data_loader is not None:
    #     lstm_model = BaselineModel(baseline_pm25 if 'baseline_pm25' in locals() else 25.0)
    #     print("ℹ️ LSTM/GRU fallback: using BaselineModel")
except Exception as e:
    print(f"⚠️ Error creating baseline fallback models: {e}")

# --- GRU output scaler (for inverse transform)
gru_y_scaler = None
if lstm_scaler is not None:  # only if GRU model is loaded
    try:
        # Try to load GRU output scaler (MinMaxScaler or StandardScaler for targets)
        gru_y_scaler_candidates = [
            os.path.join(MODEL_DIR, 'GRU', 'scalers_y.pkl'),
            os.path.join(BASE_DIR, 'Training', 'GRU', 'scalers_y.pkl'),
            os.path.join(MODEL_DIR, 'GRU', 'scaler_y_multi_conservative.pkl'),
        ]
        for cand in gru_y_scaler_candidates:
            if os.path.exists(cand):
                try:
                    gru_y_scaler = joblib.load(cand)
                    print(f"✅ GRU output scaler loaded: {cand}")
                    break
                except Exception as e:
                    print(f"⚠️ Could not load GRU output scaler from {cand}: {e}")
    except Exception as e:
        print(f"⚠️ GRU output scaler load error: {e}")

# --- Initialize engines
try:
    # PredictionEngine signature: (dt_model, dt_scaler, cnn_model, cnn_scaler, gru_model, gru_scaler, y_scaler, cnn_y_scaler, preprocessed_csv)
    # lstm_model digunakan sebagai gru_model
    preprocessed_csv_path = os.path.join(DATA_DIR, 'dataset_preprocessed', 'dataset_preprocessed.csv')
    prediction_engine = PredictionEngine(
        dt_model=dt_model,
        dt_scaler=scaler,
        cnn_model=cnn_model,
        cnn_scaler=cnn_scaler,
        cnn_y_scaler=cnn_y_scaler,  # CNN output scaler for inverse transform
        gru_model=lstm_model,  # GRU model digunakan sebagai LSTM
        gru_scaler=lstm_scaler,
        y_scaler=gru_y_scaler,  # GRU output scaler for inverse transform
        preprocessed_csv=preprocessed_csv_path if os.path.exists(preprocessed_csv_path) else None
    )
    ispu_classifier = ISPUClassifier()
    recommendation_engine = RecommendationEngine()
    print("✅ Engines initialized")
except Exception as e:
    import traceback
    print(f"❌ Engine init error: {e}")
    traceback.print_exc()
    prediction_engine = None

print("="*80)
print("✅ SERVER READY!" if dt_model else "⚠️ SERVER STARTED WITH WARNINGS")
print("="*80)

# ============================================================================
# HELPER FUNCTION
# ============================================================================
def validate_required_services():
    if prediction_engine is None:
        return False, "Prediction engine not initialized"
    if ispu_classifier is None:
        return False, "ISPU classifier not initialized"
    if data_loader is None:
        return False, "Data loader not initialized"
    return True, "OK"

# ============================================================================
# ROUTES
# ============================================================================
@app.route('/')
def home():
    return jsonify({
        'app': 'Air Quality Prediction API - Kota Malang',
        'version': '2.0.0',
        'models': {
            'decision_tree': dt_model is not None,
            'cnn': cnn_model is not None,
            'lstm': lstm_model is not None
        },
        'endpoints': [
            '/api/dashboard',
            '/api/recommendations',
            '/api/history',
            '/api/anomalies'
        ]
    })

# ----------------------------------------------------------------------------
# DASHBOARD ENDPOINT (Comparative display)
# ----------------------------------------------------------------------------
@app.route('/api/dashboard', methods=['GET'])
def dashboard():
    try:
        is_valid, msg = validate_required_services()
        if not is_valid:
            return jsonify({'status': 'error', 'message': msg}), 503

        # Handle target_date dengan error handling
        date_str = request.args.get('date')
        try:
            if date_str:
                target_date = pd.to_datetime(date_str)
            elif test_df is not None and not test_df.empty and 'tanggal' in test_df.columns:
                target_date = pd.to_datetime(test_df['tanggal'].max())
            elif train_df is not None and not train_df.empty and 'tanggal' in train_df.columns:
                target_date = pd.to_datetime(train_df['tanggal'].max())
            elif val_df is not None and not val_df.empty and 'tanggal' in val_df.columns:
                target_date = pd.to_datetime(val_df['tanggal'].max())
            else:
                # Fallback: gunakan tanggal hari ini
                from datetime import date as dt_date
                target_date = pd.to_datetime(dt_date.today())
                print("⚠️ No data found, using today's date as fallback")
        except Exception as e:
            print(f"⚠️ Error determining target_date: {e}, using today")
            from datetime import date as dt_date
            target_date = pd.to_datetime(dt_date.today())

        # Get data untuk tanggal target
        if data_loader is None:
            return jsonify({'status': 'error', 'message': 'Data loader not initialized'}), 500
        
        data_today = data_loader.get_data_for_date(target_date)
        if data_today is None:
            return jsonify({
                'status': 'error', 
                'message': f'Data not found for date {target_date.strftime("%Y-%m-%d")}',
                'suggested_date': target_date.strftime('%Y-%m-%d')
            }), 404

        # Predict with all 3 models (with error handling)
        try:
            dt_pred = prediction_engine.predict_dt(data_today)
        except Exception as e:
            print(f"⚠️ DT prediction error: {e}")
            import traceback
            traceback.print_exc()
            dt_pred = {'pm25': 25.0, 'o3': 50.0, 'co': 500.0}
        
        try:
            # Use predict_gru (GRU model, not LSTM)
            lstm_pred = prediction_engine.predict_gru(data_today)
        except Exception as e:
            print(f"⚠️ GRU/LSTM prediction error: {e}")
            import traceback
            traceback.print_exc()
            lstm_pred = {'pm25': 25.0, 'o3': 50.0, 'co': 500.0}
        
        try:
            cnn_pred = prediction_engine.predict_cnn(data_today)
        except Exception as e:
            print(f"⚠️ CNN prediction error: {e}")
            import traceback
            traceback.print_exc()
            cnn_pred = {'pm25': 25.0, 'o3': 50.0, 'co': 500.0}

        # (Confidence score feature removed) - previously calculated based on model agreement

        # Classification (with error handling)
        try:
            ispu_dt = ispu_classifier.classify_all(dt_pred.get('pm25', 25.0), dt_pred.get('o3', 50.0), dt_pred.get('co', 500.0))
        except Exception as e:
            print(f"⚠️ ISPU DT classification error: {e}")
            ispu_dt = {'overall': 'SEDANG', 'pm25': 'SEDANG', 'o3': 'SEDANG', 'co': 'SEDANG'}
        
        try:
            ispu_lstm = ispu_classifier.classify_all(lstm_pred.get('pm25', 25.0), lstm_pred.get('o3', 50.0), lstm_pred.get('co', 500.0))
        except Exception as e:
            print(f"⚠️ ISPU LSTM classification error: {e}")
            ispu_lstm = {'overall': 'SEDANG', 'pm25': 'SEDANG', 'o3': 'SEDANG', 'co': 'SEDANG'}
        
        try:
            ispu_cnn = ispu_classifier.classify_all(cnn_pred.get('pm25', 25.0), cnn_pred.get('o3', 50.0), cnn_pred.get('co', 500.0))
        except Exception as e:
            print(f"⚠️ ISPU CNN classification error: {e}")
            ispu_cnn = {'overall': 'SEDANG', 'pm25': 'SEDANG', 'o3': 'SEDANG', 'co': 'SEDANG'}

        # Get trend data from SQLite (7 days back from today)
        try:
            # Prioritize data from SQLite database
            trend_data = get_trend_data_from_db(days_back=7)
            if not trend_data:
                # Fallback to data_loader if SQLite is empty
                print("⚠️ SQLite trend data empty, trying data_loader fallback")
                trend_data = data_loader.get_trend_data(target_date, days=7) if data_loader else []
        except Exception as e:
            print(f"⚠️ Trend data error: {e}")
            # Fallback to data_loader if SQLite fails
            try:
                if data_loader:
                    trend_data = data_loader.get_trend_data(target_date, days=7)
                else:
                    trend_data = []
            except Exception as e2:
                print(f"⚠️ Fallback trend data also failed: {e2}")
                trend_data = []
        
        try:
            heatmap_data = prediction_engine.generate_heatmap(data_today)
        except Exception as e:
            print(f"⚠️ Heatmap generation error: {e}")
            heatmap_data = []
        
        # Get anomalies from SQLite database (7 latest from past data only, excluding today's prediction)
        try:
            anomalies = get_anomalies_from_db(limit=7, days_back=30)
            if not anomalies:
                # Fallback to data_loader if SQLite is empty
                print("⚠️ SQLite anomalies empty, trying data_loader fallback")
                anomalies = data_loader.get_recent_anomalies(limit=20, days_back=90) if data_loader else []
        except Exception as e:
            print(f"⚠️ Anomalies error: {e}")
            # Fallback to data_loader if SQLite fails
            try:
                if data_loader:
                    anomalies = data_loader.get_recent_anomalies(limit=20, days_back=90)
                else:
                    anomalies = []
            except Exception as e2:
                print(f"⚠️ Fallback anomalies also failed: {e2}")
                anomalies = []
        
        # Get ISPU statistics for 30 days back
        try:
            ispu_stats = get_ispu_statistics_from_db(days_back=30)
        except Exception as e:
            print(f"⚠️ ISPU statistics error: {e}")
            ispu_stats = {
                'categories': [],
                'total_days': 0,
                'total_hours': 0,
                'date_range': None
            }

        return jsonify({
            'status': 'success',
            'timestamp': target_date.strftime('%Y-%m-%d'),
            'primary_model': 'cnn',
            'predictions': {
                'decision_tree': {
                    'pm25': round(dt_pred.get('pm25', 0), 2),
                    'o3': round(dt_pred.get('o3', 0), 2),
                    'co': round(dt_pred.get('co', 0), 2),
                    'ispu': ispu_dt.get('overall', 'SEDANG')
                },
                'lstm': {
                    'pm25': round(lstm_pred.get('pm25', 0), 2),
                    'o3': round(lstm_pred.get('o3', 0), 2),
                    'co': round(lstm_pred.get('co', 0), 2),
                    'ispu': ispu_lstm.get('overall', 'SEDANG')
                },
                'cnn': {
                    'pm25': round(cnn_pred.get('pm25', 0), 2),
                    'o3': round(cnn_pred.get('o3', 0), 2),
                    'co': round(cnn_pred.get('co', 0), 2),
                    'ispu': ispu_cnn.get('overall', 'SEDANG')
                }
            },
            'trend': {
                'data': trend_data,
                'period': 'last_7_days'
            },
            'heatmap': {
                'data': heatmap_data,
                'grid_size': '10x5',
                'pollutants': ['PM2.5', 'O3', 'CO']
            },
            'anomalies': {
                'data': anomalies,
                'count': len(anomalies),
                'period': 'last_30_days'
            },
            'statistics': {
                'ispu_categories': ispu_stats.get('categories', []),
                'total_days': ispu_stats.get('total_days', 0),
                'total_hours': ispu_stats.get('total_hours', 0),
                'date_range': ispu_stats.get('date_range', None)
            }
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Dashboard error: {e}")
        print(f"❌ Traceback:\n{error_trace}")
        return jsonify({
            'status': 'error', 
            'message': str(e),
            'traceback': error_trace if app.debug else None
        }), 500

# ----------------------------------------------------------------------------
# RECOMMENDATIONS ENDPOINT (Now using CNN)
# ----------------------------------------------------------------------------
@app.route('/api/recommendations', methods=['GET'])
def recommendations():
    try:
        is_valid, msg = validate_required_services()
        if not is_valid:
            return jsonify({'status': 'error', 'message': msg}), 503

        target_date = test_df['tanggal'].max()
        data_today = data_loader.get_data_for_date(target_date)
        cnn_pred = prediction_engine.predict_cnn(data_today)
        ispu_status = ispu_classifier.classify_all(cnn_pred['pm25'], cnn_pred['o3'], cnn_pred['co'])
        recs = recommendation_engine.get_recommendations(ispu_status['overall'])

        return jsonify({
            'status': 'success',
            'date': target_date.strftime('%Y-%m-%d'),
            'model': 'cnn',
            'ispu_category': ispu_status['overall'],
            'pollutants': cnn_pred,
            'recommendations': recs
        })
    except Exception as e:
        print(f"❌ Recommendations error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ----------------------------------------------------------------------------
# HISTORY ENDPOINT (Actual measured data from database)
# ----------------------------------------------------------------------------
@app.route('/api/history', methods=['GET'])
def history():
    try:
        limit = int(request.args.get('limit', 30))
        
        # Mengambil data aktual dari database
        db_data = get_history_data(limit=limit)
        
        results = []
        if db_data:
            for record in db_data:
                # Hitung ISPU berdasarkan data aktual
                # Pastikan nilai tidak None (bisa terjadi jika data kosong)
                pm25 = record['pm25'] or 0
                o3 = record['o3'] or 0
                co = record['co'] or 0
                
                ispu = ispu_classifier.classify_all(pm25, o3, co)
                
                results.append({
                    'date': str(record['date']),
                    'data_type': 'actual',  # Menandakan ini adalah data aktual
                    'measurements': {
                        'pm25': pm25,
                        'o3': o3,
                        'co': co
                    },
                    'ispu': ispu
                })
        else:
            # Fallback jika DB kosong (opsional, atau return kosong)
            pass
            
        return jsonify({'status': 'success', 'records': results})
    except Exception as e:
        print(f"❌ History error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/history/export', methods=['GET'])
def history_export():
    try:
        limit = int(request.args.get('limit', 20)) # Default to 20 records (last 20 days)
        db_data = get_history_data(limit=limit)
        
        if not db_data:
            return jsonify({'status': 'error', 'message': 'No data available to export'}), 404
            
        # Prepare data for DataFrame with actual measurements
        export_data = []
        for record in db_data:
            pm25 = record['pm25'] or 0
            o3 = record['o3'] or 0
            co = record['co'] or 0
            ispu = ispu_classifier.classify_all(pm25, o3, co)
            
            export_data.append({
                'Tanggal': record['date'],
                'PM2.5 (µg/m³)': pm25,
                'O3 (µg/m³)': o3,
                'CO (µg/m³)': co,
                'Data Type': 'Actual',
                'ISPU Category': ispu['overall'],
                'ISPU Advice': ispu['advice']
            })
            
        df = pd.DataFrame(export_data)
        
        # Save to buffer
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='History')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f"air_quality_history_{datetime.now().strftime('%Y%m%d')}.xlsx"
        )
        
    except Exception as e:
        print(f"❌ Export error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ----------------------------------------------------------------------------
# ANOMALIES ENDPOINT
# ----------------------------------------------------------------------------
@app.route('/api/anomalies', methods=['GET'])
def anomalies():
    try:
        limit = int(request.args.get('limit', 50))
        # Prioritize data from SQLite database
        anomaly_list = get_anomalies_from_db(limit=limit, days_back=90)
        if not anomaly_list:
            # Fallback to data_loader if SQLite is empty
            anomaly_list = data_loader.get_recent_anomalies(limit=limit) if data_loader else []
        return jsonify({
            'status': 'success',
            'total_anomalies': len(anomaly_list),
            'anomalies': anomaly_list
        })
    except Exception as e:
        print(f"❌ Anomalies error: {e}")
        # Fallback to data_loader if SQLite fails
        try:
            if data_loader:
                limit = int(request.args.get('limit', 10))
                anomaly_list = data_loader.get_recent_anomalies(limit=limit)
                return jsonify({
                    'status': 'success',
                    'total_anomalies': len(anomaly_list),
                    'anomalies': anomaly_list
                })
        except Exception as e2:
            pass
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============================================================================
# MANUAL CRAWL & TRAIN ENDPOINT
# ============================================================================
@app.route('/api/crawl-now', methods=['POST'])
def crawl_now():
    """
    Endpoint untuk manual trigger crawling dan training sekarang (tidak perlu menunggu 23:50).
    Berguna untuk testing dan backfill data.
    """
    try:
        from crawler.scheduler import run_crawl_and_train, crawl_daily
        
        crawl_only = request.json.get('crawl_only', False) if request.json else False
        
        if crawl_only:
            # Hanya crawl, tidak train
            print("📥 Starting manual crawl (without training)...")
            result = crawl_daily()
            if result:
                return jsonify({
                    'status': 'success',
                    'message': 'Crawling completed successfully',
                    'mode': 'crawl_only'
                }), 200
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Crawling failed'
                }), 500
        else:
            # Crawl + Train
            print("📥 Starting manual crawl + train...")
            result = run_crawl_and_train()
            if result:
                return jsonify({
                    'status': 'success',
                    'message': 'Crawling and training completed successfully',
                    'mode': 'crawl_and_train'
                }), 200
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Crawl and train process failed'
                }), 500
    except Exception as e:
        print(f"❌ Manual crawl error: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error during crawl: {str(e)}'
        }), 500

# ============================================================================
# START SCHEDULER (CONTINUOUS LEARNING) - OPSI 3
# ============================================================================
if SCHEDULER_AVAILABLE:
    try:
        logger.info("\n" + "="*80)
        logger.info("� STARTING CONTINUOUS LEARNING SCHEDULER (OPSI 3)")
        logger.info("="*80)
        
        scheduler = start_scheduler()
        
        if scheduler:
            logger.info("✅ CONTINUOUS LEARNING SCHEDULER STARTED!")
            logger.info("   HOURLY: Crawl data setiap jam")
            logger.info("   DAILY: Aggregation + Training setiap hari jam 23:50 WIB")
            logger.info("="*80)
            logger.info("Log file: backend/logs/continuous_learning.log\n")
        else:
            logger.warning("⚠️ Scheduler gagal dimulai (APScheduler mungkin belum terinstall)")
            logger.warning("   Install dengan: pip install apscheduler\n")
    except Exception as e:
        logger.error(f"❌ Error starting scheduler: {e}", exc_info=True)
        logger.warning("   Continuous learning tidak aktif, model tidak akan auto-update\n")
else:
    logger.warning("⚠️ Scheduler module tidak tersedia")
    logger.warning("   Continuous learning tidak aktif\n")

# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 2000))
    logger.info(f"\n🌐 Server running at http://127.0.0.1:{port}")
    logger.info("🎯 Backend API ready for dashboard connection\n")
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)

