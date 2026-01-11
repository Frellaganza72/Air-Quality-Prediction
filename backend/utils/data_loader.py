"""
================================================================================
DATA LOADER - Load and manage datasets
================================================================================
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import joblib

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.outliers_df = None
        self.X_scaler = None  # MinMaxScaler for weather features
        self._load_X_scaler()
        
    def _load_X_scaler(self):
        """Load the MinMaxScaler used for normalizing weather features during training"""
        try:
            # Try to load the GRU feature scaler which contains the MinMaxScaler for weather data
            scaler_path = os.path.join(self.data_dir, '..', 'models', 'GRU', 'scaler_X.pkl')
            if os.path.exists(scaler_path):
                self.X_scaler = joblib.load(scaler_path)
                return
            
            # Also try the old location
            scaler_path = os.path.join(self.data_dir, 'scaler_X.pkl')
            if os.path.exists(scaler_path):
                self.X_scaler = joblib.load(scaler_path)
                return
        except Exception as e:
            pass  # X_scaler remains None if loading fails

    def load_train(self):
        """Load training dataset"""
        try:
            # Try multiple locations in order of preference
            possible_paths = [
                os.path.join(self.data_dir, 'train.csv'),  # Backend/data/train.csv
                os.path.join(self.data_dir, '..', 'Training', 'shared_data', 'train.csv'),  # Backend/Training/shared_data/train.csv
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df['tanggal'] = pd.to_datetime(df['tanggal'])
                    self.train_df = df
                    print(f"✅ Loaded train data from: {path}")
                    return df
            
            # Try loading from dataset_preprocessed
            preprocessed_paths = [
                os.path.join(self.data_dir, 'dataset_preprocessed', 'dataset_preprocessed.csv'),
                os.path.join(self.data_dir, '..', 'Training', 'shared_data', 'dataset_preprocessed.csv'),
            ]
            for preprocessed_path in preprocessed_paths:
                if os.path.exists(preprocessed_path):
                    df = pd.read_csv(preprocessed_path)
                    df['tanggal'] = pd.to_datetime(df['tanggal'])
                    # Use first 60% as train data
                    train_size = int(len(df) * 0.6)
                    self.train_df = df.head(train_size).reset_index(drop=True)
                    print(f"✅ Loaded train data from preprocessed dataset: {len(self.train_df)} rows")
                    return self.train_df
            
            print(f"⚠️ Train data not found in any location")
            return self._create_dummy_data(500)
        except Exception as e:
            print(f"❌ Error loading train data: {e}")
            return self._create_dummy_data(500)
    
    def load_val(self):
        """Load validation dataset"""
        try:
            # Try multiple locations in order of preference
            possible_paths = [
                os.path.join(self.data_dir, 'val.csv'),  # Backend/data/val.csv
                os.path.join(self.data_dir, '..', 'Training', 'shared_data', 'val.csv'),  # Backend/Training/shared_data/val.csv
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df['tanggal'] = pd.to_datetime(df['tanggal'])
                    self.val_df = df
                    print(f"✅ Loaded validation data from: {path}")
                    return df
            
            # Try loading from dataset_preprocessed
            preprocessed_paths = [
                os.path.join(self.data_dir, 'dataset_preprocessed', 'dataset_preprocessed.csv'),
                os.path.join(self.data_dir, '..', 'Training', 'shared_data', 'dataset_preprocessed.csv'),
            ]
            for preprocessed_path in preprocessed_paths:
                if os.path.exists(preprocessed_path):
                    df = pd.read_csv(preprocessed_path)
                    df['tanggal'] = pd.to_datetime(df['tanggal'])
                    # Use middle 20% as validation data (60% to 80%)
                    train_size = int(len(df) * 0.6)
                    val_size = int(len(df) * 0.2)
                    self.val_df = df.iloc[train_size:train_size + val_size].reset_index(drop=True)
                    print(f"✅ Loaded validation data from preprocessed dataset: {len(self.val_df)} rows")
                    return self.val_df
            
            print(f"⚠️ Validation data not found in any location")
            return self._create_dummy_data(100)
        except Exception as e:
            print(f"❌ Error loading validation data: {e}")
            return self._create_dummy_data(100)
    
    def load_test(self):
        """Load test dataset"""
        try:
            # Try multiple locations in order of preference
            possible_paths = [
                os.path.join(self.data_dir, 'test.csv'),  # Backend/data/test.csv
                os.path.join(self.data_dir, '..', 'Training', 'shared_data', 'test.csv'),  # Backend/Training/shared_data/test.csv
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df['tanggal'] = pd.to_datetime(df['tanggal'])
                    self.test_df = df
                    print(f"✅ Loaded test data from: {path}")
                    return df
            
            # Try loading from dataset_preprocessed
            preprocessed_paths = [
                os.path.join(self.data_dir, 'dataset_preprocessed', 'dataset_preprocessed.csv'),
                os.path.join(self.data_dir, '..', 'Training', 'shared_data', 'dataset_preprocessed.csv'),
            ]
            for preprocessed_path in preprocessed_paths:
                if os.path.exists(preprocessed_path):
                    df = pd.read_csv(preprocessed_path)
                    df['tanggal'] = pd.to_datetime(df['tanggal'])
                    # Use last 20% as test data
                    test_size = int(len(df) * 0.2)
                    self.test_df = df.tail(test_size).reset_index(drop=True)
                    print(f"✅ Loaded test data from preprocessed dataset: {len(self.test_df)} rows")
                    return self.test_df
            
            print(f"⚠️ Test data not found in any location")
            return self._create_dummy_data(100)
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            return self._create_dummy_data(100)
    
    def load_outliers(self):
        """Load outliers/anomalies dataset"""
        try:
            path = os.path.join(self.data_dir, 'outliers.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['tanggal'] = pd.to_datetime(df['tanggal'])
                self.outliers_df = df
                return df
            else:
                print(f"⚠️ Outliers data not found at {path}")
                return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error loading outliers data: {e}")
            return pd.DataFrame()
    
    def _create_dummy_data(self, n_samples):
        """Create dummy data for testing when real data is not available"""
        dates = pd.date_range(
            end=datetime.now(),
            periods=n_samples,
            freq='3H'  # 3-hour intervals
        )
        
        # Simulate realistic patterns
        np.random.seed(42)
        
        df = pd.DataFrame({
            'tanggal': dates,
            'pm2_5 (μg/m³)_mean': np.random.uniform(5, 45, n_samples),
            'ozone (μg/m³)_mean': np.random.uniform(20, 120, n_samples),
            'carbon_monoxide (μg/m³)_mean': np.random.uniform(200, 4000, n_samples),
            'temperature_2m (°C)_mean': np.random.uniform(22, 32, n_samples),
            'relative_humidity_2m (%)_mean': np.random.uniform(60, 95, n_samples),
            'wind_speed_10m (km/h)_mean': np.random.uniform(2, 20, n_samples),
            'surface_pressure (hPa)_mean': np.random.uniform(1010, 1020, n_samples),
            'precipitation (mm)_sum': np.random.uniform(0, 10, n_samples)
        })
        
        return df
    
    def get_latest_data_from_sqlite(self, target_date=None):
        """
        Fetch the most recent day's aggregated data from SQLite database.
        Used as fallback when requesting future dates (e.g., today when no data exists yet).
        
        If target_date is provided, use its temporal features (hari, bulan) instead of latest date's.
        This allows different predictions for different dates even when using the same pollutant values.
        """
        try:
            import sqlite3
            db_path = os.path.join(self.data_dir, 'datacrawler.db')
            if not os.path.exists(db_path):
                return None
            
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get the latest date with data
            cursor.execute("SELECT MAX(tanggal) as latest FROM polutan")
            result = cursor.fetchone()
            
            if not result or not result['latest']:
                conn.close()
                return None
            
            latest_date_str = result['latest']
            
            # Fetch aggregated pollutant data for that day
            cursor.execute("""
                SELECT tanggal, 
                       pm2_5_mean, pm2_5_max, pm2_5_min, pm2_5_median,
                       ozone_mean, ozone_max, ozone_min, ozone_median,
                       carbon_monoxide_mean, carbon_monoxide_max, carbon_monoxide_min, carbon_monoxide_median,
                       pm10_mean, pm10_max, pm10_min, pm10_median
                FROM polutan 
                WHERE tanggal = ?
            """, (latest_date_str,))
            
            polutan_row = cursor.fetchone()
            if polutan_row is None:
                conn.close()
                return None
            
            # Fetch aggregated weather data for that day
            cursor.execute("""
                SELECT tanggal,
                       temp_mean, temp_max, tempmax, tempmin,
                       humidity_mean, humidity_max,
                       windspeed_mean, windspeed_max, winddir_mean,
                       sealevelpressure_mean, sealevelpressure_max,
                       cloudcover_mean, cloudcover_max,
                       visibility_mean, visibility_max,
                       solarradiation_mean, solarradiation_max
                FROM cuaca 
                WHERE tanggal = ?
            """, (latest_date_str,))
            
            weather_row = cursor.fetchone()
            conn.close()
            
            # Build aggregated data using database values
            agg_data = {
                'tanggal': pd.to_datetime(latest_date_str),
                'pm2_5 (μg/m³)_mean': float(polutan_row['pm2_5_mean']) if polutan_row['pm2_5_mean'] is not None else 25.0,
                'pm2_5 (μg/m³)_max': float(polutan_row['pm2_5_max']) if polutan_row['pm2_5_max'] is not None else 40.0,
                'pm2_5 (μg/m³)_median': float(polutan_row['pm2_5_median']) if polutan_row['pm2_5_median'] is not None else 25.0,
                'ozone (μg/m³)_mean': float(polutan_row['ozone_mean']) if polutan_row['ozone_mean'] is not None else 50.0,
                'ozone (μg/m³)_max': float(polutan_row['ozone_max']) if polutan_row['ozone_max'] is not None else 80.0,
                'ozone (μg/m³)_median': float(polutan_row['ozone_median']) if polutan_row['ozone_median'] is not None else 50.0,
                'carbon_monoxide (μg/m³)_mean': float(polutan_row['carbon_monoxide_mean']) if polutan_row['carbon_monoxide_mean'] is not None else 500.0,
                'carbon_monoxide (μg/m³)_max': float(polutan_row['carbon_monoxide_max']) if polutan_row['carbon_monoxide_max'] is not None else 800.0,
                'carbon_monoxide (μg/m³)_median': float(polutan_row['carbon_monoxide_median']) if polutan_row['carbon_monoxide_median'] is not None else 500.0,
            }
            
            # Add weather data if available
            if weather_row is not None:
                agg_data['tempmin'] = float(weather_row['tempmin']) if weather_row['tempmin'] is not None else 20.0
                agg_data['tempmax'] = float(weather_row['tempmax']) if weather_row['tempmax'] is not None else 30.0
                agg_data['temp'] = float(weather_row['temp_mean']) if weather_row['temp_mean'] is not None else 25.0
                agg_data['humidity'] = float(weather_row['humidity_mean']) if weather_row['humidity_mean'] is not None else 80.0
                agg_data['windspeed'] = float(weather_row['windspeed_mean']) if weather_row['windspeed_mean'] is not None else 5.0
                agg_data['winddir'] = float(weather_row['winddir_mean']) if weather_row['winddir_mean'] is not None else 180.0
                agg_data['sealevelpressure'] = float(weather_row['sealevelpressure_mean']) if weather_row['sealevelpressure_mean'] is not None else 1013.0
                agg_data['cloudcover'] = float(weather_row['cloudcover_mean']) if weather_row['cloudcover_mean'] is not None else 50.0
                agg_data['visibility'] = float(weather_row['visibility_mean']) if weather_row['visibility_mean'] is not None else 10.0
                agg_data['solarradiation'] = float(weather_row['solarradiation_mean']) if weather_row['solarradiation_mean'] is not None else 200.0
            
            # Add temporal features
            # If target_date provided, use its temporal info; otherwise use latest date's
            date_for_temporal = target_date if target_date is not None else pd.to_datetime(latest_date_str)
            agg_data['hari'] = date_for_temporal.day
            agg_data['bulan'] = date_for_temporal.month
            agg_data['is_weekend'] = 1 if date_for_temporal.weekday() >= 5 else 0
            agg_data['tanggal'] = date_for_temporal
            
            # Normalize weather features
            normalization_ranges = {
                'tempmax': (15, 35),
                'tempmin': (15, 30),
                'temp': (15, 35),
                'humidity': (40, 95),
                'windspeed': (0, 30),
                'winddir': (0, 360),
                'sealevelpressure': (1010, 1015),
                'cloudcover': (0, 100),
                'visibility': (0, 50),
                'solarradiation': (0, 800),
            }
            
            for feature, (min_val, max_val) in normalization_ranges.items():
                if feature in agg_data and agg_data[feature] is not None:
                    raw_val = agg_data[feature]
                    if max_val > min_val:
                        normalized = (raw_val - min_val) / (max_val - min_val)
                        agg_data[feature] = np.clip(normalized, 0.0, 1.0)
                    else:
                        agg_data[feature] = 0.5
            
            return agg_data
            
        except Exception as e:
            print(f"⚠️ Error fetching latest data from SQLite: {e}")
            import traceback
            traceback.print_exc()
            return None
            return None
    
    def get_data_for_date_from_sqlite(self, target_date):
        """
        Fetch aggregated daily data from SQLite database for a specific date.
        Data is already aggregated in database (mean, max, min, median).
        """
        try:
            import sqlite3
            db_path = os.path.join(self.data_dir, 'datacrawler.db')
            if not os.path.exists(db_path):
                return None
            
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            date_str = target_date.strftime('%Y-%m-%d')
            
            # Fetch aggregated pollutant data (already computed in DB)
            cursor.execute("""
                SELECT tanggal, 
                       pm2_5_mean, pm2_5_max, pm2_5_min, pm2_5_median,
                       ozone_mean, ozone_max, ozone_min, ozone_median,
                       carbon_monoxide_mean, carbon_monoxide_max, carbon_monoxide_min, carbon_monoxide_median,
                       pm10_mean, pm10_max, pm10_min, pm10_median
                FROM polutan 
                WHERE tanggal = ?
            """, (date_str,))
            
            polutan_row = cursor.fetchone()
            if polutan_row is None:
                conn.close()
                return None
            
            # Fetch aggregated weather data (already computed in DB)
            cursor.execute("""
                SELECT tanggal,
                       temp_mean, temp_max, tempmax, tempmin,
                       humidity_mean, humidity_max,
                       windspeed_mean, windspeed_max, winddir_mean,
                       sealevelpressure_mean, sealevelpressure_max,
                       cloudcover_mean, cloudcover_max,
                       visibility_mean, visibility_max,
                       solarradiation_mean, solarradiation_max
                FROM cuaca 
                WHERE tanggal = ?
            """, (date_str,))
            
            weather_row = cursor.fetchone()
            conn.close()
            
            # Build aggregated data dictionary using database values
            agg_data = {
                'tanggal': target_date,
                'pm2_5 (μg/m³)_mean': float(polutan_row['pm2_5_mean']) if polutan_row['pm2_5_mean'] is not None else 25.0,
                'pm2_5 (μg/m³)_max': float(polutan_row['pm2_5_max']) if polutan_row['pm2_5_max'] is not None else 40.0,
                'pm2_5 (μg/m³)_median': float(polutan_row['pm2_5_median']) if polutan_row['pm2_5_median'] is not None else 25.0,
                'ozone (μg/m³)_mean': float(polutan_row['ozone_mean']) if polutan_row['ozone_mean'] is not None else 50.0,
                'ozone (μg/m³)_max': float(polutan_row['ozone_max']) if polutan_row['ozone_max'] is not None else 80.0,
                'ozone (μg/m³)_median': float(polutan_row['ozone_median']) if polutan_row['ozone_median'] is not None else 50.0,
                'carbon_monoxide (μg/m³)_mean': float(polutan_row['carbon_monoxide_mean']) if polutan_row['carbon_monoxide_mean'] is not None else 500.0,
                'carbon_monoxide (μg/m³)_max': float(polutan_row['carbon_monoxide_max']) if polutan_row['carbon_monoxide_max'] is not None else 800.0,
                'carbon_monoxide (μg/m³)_median': float(polutan_row['carbon_monoxide_median']) if polutan_row['carbon_monoxide_median'] is not None else 500.0,
            }
            
            # Add weather data if available
            if weather_row is not None:
                agg_data['tempmin'] = float(weather_row['tempmin']) if weather_row['tempmin'] is not None else 20.0
                agg_data['tempmax'] = float(weather_row['tempmax']) if weather_row['tempmax'] is not None else 30.0
                agg_data['temp'] = float(weather_row['temp_mean']) if weather_row['temp_mean'] is not None else 25.0
                agg_data['humidity'] = float(weather_row['humidity_mean']) if weather_row['humidity_mean'] is not None else 80.0
                agg_data['windspeed'] = float(weather_row['windspeed_mean']) if weather_row['windspeed_mean'] is not None else 5.0
                agg_data['winddir'] = float(weather_row['winddir_mean']) if weather_row['winddir_mean'] is not None else 180.0
                agg_data['sealevelpressure'] = float(weather_row['sealevelpressure_mean']) if weather_row['sealevelpressure_mean'] is not None else 1013.0
                agg_data['cloudcover'] = float(weather_row['cloudcover_mean']) if weather_row['cloudcover_mean'] is not None else 50.0
                agg_data['visibility'] = float(weather_row['visibility_mean']) if weather_row['visibility_mean'] is not None else 10.0
                agg_data['solarradiation'] = float(weather_row['solarradiation_mean']) if weather_row['solarradiation_mean'] is not None else 200.0
            
            # Add temporal features
            agg_data['hari'] = target_date.day
            agg_data['bulan'] = target_date.month
            agg_data['is_weekend'] = 1 if target_date.weekday() >= 5 else 0
            
            # Normalize weather features to [0, 1] range to match training data
            normalization_ranges = {
                'tempmax': (15, 35),
                'tempmin': (15, 30),
                'temp': (15, 35),
                'humidity': (40, 95),
                'windspeed': (0, 30),
                'winddir': (0, 360),
                'sealevelpressure': (1010, 1015),
                'cloudcover': (0, 100),
                'visibility': (0, 50),
                'solarradiation': (0, 800),
            }
            
            for feature, (min_val, max_val) in normalization_ranges.items():
                if feature in agg_data and agg_data[feature] is not None:
                    raw_val = agg_data[feature]
                    if max_val > min_val:
                        normalized = (raw_val - min_val) / (max_val - min_val)
                        agg_data[feature] = np.clip(normalized, 0.0, 1.0)
                    else:
                        agg_data[feature] = 0.5
            
            return agg_data
            
        except Exception as e:
            print(f"⚠️ Error fetching from SQLite: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_data_for_date(self, target_date):
        """
        Get data for a specific date.
        Priority: SQLite (real crawler data) > Latest available SQLite (with target date's temporal features) > Preprocessed CSV > None
        
        Special case: If requesting a date with no data (e.g., today when current day has no data yet),
        use the latest available date's pollutant/weather data but with target date's temporal features (hari, bulan).
        This ensures predictions differ day-to-day even when actual pollutant measurements are the same.
        """
        # First, try to fetch from SQLite database (real crawler data) for the exact date
        sqlite_data = self.get_data_for_date_from_sqlite(target_date)
        if sqlite_data is not None:
            return sqlite_data
        
        # If no exact date found in SQLite, check if this is a "future" date (today or later)
        # In that case, use the latest available data with target date's temporal features
        try:
            import sqlite3
            db_path = os.path.join(self.data_dir, 'datacrawler.db')
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(tanggal) FROM polutan")
                result = cursor.fetchone()
                conn.close()
                
                if result and result[0]:
                    latest_db_date = pd.to_datetime(result[0]).date()
                    target_date_only = target_date.date()
                    
                    # If target date is after latest data in DB, use latest data with target temporal features
                    if target_date_only > latest_db_date:
                        latest_data = self.get_latest_data_from_sqlite(target_date=target_date)
                        if latest_data is not None:
                            return latest_data
        except Exception as e:
            print(f"⚠️ Error checking SQLite for latest date: {e}")
        
        # Fallback: Search in test set first, then val, then train (preprocessed data)
        for df in [self.test_df, self.val_df, self.train_df]:
            if df is not None and len(df) > 0:
                # Try exact date match
                mask = df['tanggal'].dt.date == target_date.date()
                if mask.any():
                    return df[mask].iloc[0].to_dict()
                
                # Try closest date within 3 days
                df_copy = df.copy()
                df_copy['date_diff'] = abs(df_copy['tanggal'] - target_date)
                closest = df_copy.nsmallest(1, 'date_diff')
                
                if len(closest) > 0 and closest['date_diff'].iloc[0] < timedelta(days=3):
                    return closest.iloc[0].to_dict()
        
        return None
    
    def get_trend_data(self, target_date, days=7):
        """Get trend data for the past N days"""
        start_date = target_date - timedelta(days=days)
        
        # Combine all datasets
        all_data = []
        for df in [self.train_df, self.val_df, self.test_df]:
            if df is not None:
                all_data.append(df)
        
        if not all_data:
            return []
        
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values('tanggal')
        
        # Filter date range
        mask = (combined['tanggal'] >= start_date) & (combined['tanggal'] <= target_date)
        filtered = combined[mask].copy()  # Use .copy() to avoid SettingWithCopyWarning
        
        # Group by date and calculate daily averages
        filtered['date_only'] = filtered['tanggal'].dt.date
        daily_avg = filtered.groupby('date_only').agg({
            'pm2_5 (μg/m³)_mean': 'mean',
            'ozone (μg/m³)_mean': 'mean',
            'carbon_monoxide (μg/m³)_mean': 'mean'
        }).reset_index()
        
        # Format for API response
        trend = []
        for _, row in daily_avg.iterrows():
            trend.append({
                'date': str(row['date_only']),
                'pm25': round(row['pm2_5 (μg/m³)_mean'], 2),
                'o3': round(row['ozone (μg/m³)_mean'], 2),
                'co': round(row['carbon_monoxide (μg/m³)_mean'], 2)
            })
        
        return trend
    
    def get_recent_anomalies(self, limit=50, days_back=365):
        """Get recent anomaly events (latest per month)"""
        if self.outliers_df is None or len(self.outliers_df) == 0:
            return []
        
        # Filter by date (last N days - default 1 year to show monthly trends)
        cutoff_date = datetime.now() - timedelta(days=days_back)
        mask = self.outliers_df['tanggal'] >= cutoff_date
        recent_outliers = self.outliers_df[mask].copy()
        
        if recent_outliers.empty:
            return []

        # Tambahkan kolom bulan
        recent_outliers['month_year'] = recent_outliers['tanggal'].dt.to_period('M')
        
        # Sort ascending agar saat grouping 'last' kita dapat tanggal terakhir di bulan itu
        recent_outliers = recent_outliers.sort_values('tanggal')
        
        # Filter hanya yang benar-benar anomali (melewati threshold)
        # Thresholds: PM2.5 > 35, O3 > 120, CO > 4000
        is_anomaly = (
            (recent_outliers['pm2_5 (μg/m³)_mean'] > 35) | 
            (recent_outliers['ozone (μg/m³)_mean'] > 120) | 
            (recent_outliers['carbon_monoxide (μg/m³)_mean'] > 4000)
        )
        recent_outliers = recent_outliers[is_anomaly]
        
        if recent_outliers.empty:
            return []
            
        # Group by month, take the LAST one (latest date in that month)
        monthly_outliers = recent_outliers.groupby('month_year').last().reset_index()
        
        # Sort descending untuk output (bulan terbaru di atas)
        sorted_outliers = monthly_outliers.sort_values('tanggal', ascending=False)
        
        anomalies = []
        for _, row in sorted_outliers.head(limit).iterrows():
            # Determine which pollutant is anomalous
            pm25 = row['pm2_5 (μg/m³)_mean']
            o3 = row['ozone (μg/m³)_mean']
            co = row['carbon_monoxide (μg/m³)_mean']
            
            # Identify the primary cause
            # Normalize to threshold to find max violator
            pm25_ratio = pm25 / 35.0
            o3_ratio = o3 / 120.0
            co_ratio = co / 4000.0
            
            pollutant = None
            value = 0.0
            baseline = 0.0
            increase = 0.0
            
            max_ratio = max(pm25_ratio, o3_ratio, co_ratio)
            
            if max_ratio > 1.0:
                if max_ratio == pm25_ratio:
                    pollutant = 'PM2.5'
                    value = pm25
                    baseline = 15.5
                    increase = ((pm25 - baseline) / baseline * 100)
                elif max_ratio == o3_ratio:
                    pollutant = 'O3'
                    value = o3
                    baseline = 60
                    increase = ((o3 - baseline) / baseline * 100)
                else:
                    pollutant = 'CO'
                    value = co
                    baseline = 2000
                    increase = ((co - baseline) / baseline * 100)
            
                anomalies.append({
                    'datetime': row['tanggal'].strftime('%Y-%m-%d %H:%M'),
                    'pollutant': pollutant,
                    'value': round(value, 2),
                    'increase_percent': round(increase, 1),
                    'description': f"Lonjakan {pollutant}: {round(value, 1)} (↑{round(increase, 1)}%)"
                })
        
        return anomalies
    
    def get_all_data(self):
        """Get all available data combined"""
        all_data = []
        for df in [self.train_df, self.val_df, self.test_df]:
            if df is not None:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True).sort_values('tanggal')
    
    def get_latest_data(self):
        """Get the most recent data point"""
        all_data = self.get_all_data()
        if len(all_data) == 0:
            return None
        
        return all_data.iloc[-1].to_dict()
    
    def get_spatial_grid(self, target_date):
        """
        Get spatial grid data for heatmap
        Returns 15x15 grid representing different areas in Malang
        """
        # In production, this would query actual spatial data
        # For now, we'll generate a grid based on temporal variation
        
        data = self.get_data_for_date(target_date)
        if data is None:
            return None
        
        # Create 15x15 grid with variations around the mean value
        base_pm25 = data['pm2_5 (μg/m³)_mean']
        
        # Add spatial variation (simulate different districts)
        np.random.seed(int(target_date.timestamp()))
        grid = np.random.normal(base_pm25, base_pm25 * 0.2, (15, 15))
        
        # Ensure non-negative values
        grid = np.clip(grid, 0, None)
        
        return grid