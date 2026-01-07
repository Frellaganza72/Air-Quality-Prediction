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
        Fetch and aggregate the most recent day's data from SQLite database.
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
            cursor.execute("""
                SELECT DISTINCT tanggal FROM polutan 
                ORDER BY tanggal DESC LIMIT 1
            """)
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                return None
            
            latest_date_str = result['tanggal']
            
            # Fetch pollutant data for that day
            cursor.execute("""
                SELECT waktu, pm2_5, ozone, carbon_monoxide 
                FROM polutan 
                WHERE tanggal = ? 
                ORDER BY waktu
            """, (latest_date_str,))
            
            polutan_rows = cursor.fetchall()
            if not polutan_rows:
                conn.close()
                return None
            
            # Fetch weather data for that day
            cursor.execute("""
                SELECT datetime, temp, humidity, windspeed, winddir, 
                       sealevelpressure, cloudcover, visibility, solarradiation
                FROM cuaca 
                WHERE tanggal = ? 
                ORDER BY datetime
            """, (latest_date_str,))
            
            weather_rows = cursor.fetchall()
            conn.close()
            
            # Aggregate pollutant data
            pm25_values = [float(row['pm2_5']) for row in polutan_rows if row['pm2_5'] is not None]
            o3_values = [float(row['ozone']) for row in polutan_rows if row['ozone'] is not None]
            co_values = [float(row['carbon_monoxide']) for row in polutan_rows if row['carbon_monoxide'] is not None]
            
            if not pm25_values or not o3_values or not co_values:
                return None
            
            agg_data = {
                'tanggal': pd.to_datetime(latest_date_str),
                'pm2_5 (μg/m³)_mean': np.mean(pm25_values),
                'pm2_5 (μg/m³)_max': np.max(pm25_values),
                'pm2_5 (μg/m³)_median': np.median(pm25_values),
                'ozone (μg/m³)_mean': np.mean(o3_values),
                'ozone (μg/m³)_max': np.max(o3_values),
                'ozone (μg/m³)_median': np.median(o3_values),
                'carbon_monoxide (μg/m³)_mean': np.mean(co_values),
                'carbon_monoxide (μg/m³)_max': np.max(co_values),
                'carbon_monoxide (μg/m³)_median': np.median(co_values),
            }
            
            # Aggregate weather data
            if weather_rows:
                temp_values = [float(row['temp']) for row in weather_rows if row['temp'] is not None]
                humidity_values = [float(row['humidity']) for row in weather_rows if row['humidity'] is not None]
                windspeed_values = [float(row['windspeed']) for row in weather_rows if row['windspeed'] is not None]
                winddir_values = [float(row['winddir']) for row in weather_rows if row['winddir'] is not None]
                pressure_values = [float(row['sealevelpressure']) for row in weather_rows if row['sealevelpressure'] is not None]
                cloudcover_values = [float(row['cloudcover']) for row in weather_rows if row['cloudcover'] is not None]
                visibility_values = [float(row['visibility']) for row in weather_rows if row['visibility'] is not None]
                solarrad_values = [float(row['solarradiation']) for row in weather_rows if row['solarradiation'] is not None]
                
                if temp_values:
                    agg_data['tempmin'] = np.min(temp_values)
                    agg_data['tempmax'] = np.max(temp_values)
                    agg_data['temp'] = np.mean(temp_values)
                if humidity_values:
                    agg_data['humidity'] = np.mean(humidity_values)
                if windspeed_values:
                    agg_data['windspeed'] = np.mean(windspeed_values)
                if winddir_values:
                    agg_data['winddir'] = np.mean(winddir_values)
                if pressure_values:
                    agg_data['sealevelpressure'] = np.mean(pressure_values)
                if cloudcover_values:
                    agg_data['cloudcover'] = np.mean(cloudcover_values)
                if visibility_values:
                    agg_data['visibility'] = np.mean(visibility_values)
                if solarrad_values:
                    agg_data['solarradiation'] = np.mean(solarrad_values)
            
            # Add temporal features
            # If target_date provided, use its temporal info; otherwise use latest date's
            date_for_temporal = target_date if target_date is not None else pd.to_datetime(latest_date_str)
            agg_data['hari'] = date_for_temporal.day
            agg_data['bulan'] = date_for_temporal.month
            agg_data['is_weekend'] = 1 if date_for_temporal.weekday() >= 5 else 0
            agg_data['tanggal'] = date_for_temporal  # Override with target date for consistency
            
            # Normalize weather features to [0, 1] range to match training data
            # Use typical min/max ranges for Indonesian tropical weather
            normalization_ranges = {
                'tempmax': (15, 35),      # Typical max temp in Celsius
                'tempmin': (15, 30),      # Typical min temp
                'temp': (15, 35),         # Typical mean temp
                'humidity': (40, 95),     # Typical humidity percentage
                'windspeed': (0, 30),     # Typical wind speed km/h
                'winddir': (0, 360),      # Wind direction degrees
                'sealevelpressure': (1010, 1015),  # hPa
                'cloudcover': (0, 100),   # Cloud cover percentage
                'visibility': (0, 50),    # km
                'solarradiation': (0, 800),  # W/m2
            }
            
            for feature, (min_val, max_val) in normalization_ranges.items():
                if feature in agg_data and agg_data[feature] is not None:
                    # Normalize to [0, 1]
                    raw_val = agg_data[feature]
                    if max_val > min_val:
                        normalized = (raw_val - min_val) / (max_val - min_val)
                        # Clip to [0, 1] range
                        agg_data[feature] = np.clip(normalized, 0.0, 1.0)
                    else:
                        agg_data[feature] = 0.5  # fallback to middle value
            
            return agg_data
            
        except Exception as e:
            print(f"⚠️ Error fetching latest data from SQLite: {e}")
            return None
    
    def get_data_for_date_from_sqlite(self, target_date):
        """
        Fetch and aggregate hourly data from SQLite database for a specific date.
        Returns daily aggregated data with mean/max/median for pollutants and weather.
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
            
            # Fetch pollutant data for the day
            cursor.execute("""
                SELECT waktu, pm2_5, ozone, carbon_monoxide 
                FROM polutan 
                WHERE tanggal = ? 
                ORDER BY waktu
            """, (date_str,))
            
            polutan_rows = cursor.fetchall()
            if not polutan_rows:
                conn.close()
                return None
            
            # Fetch weather data for the day
            cursor.execute("""
                SELECT datetime, temp, humidity, windspeed, winddir, 
                       sealevelpressure, cloudcover, visibility, solarradiation
                FROM cuaca 
                WHERE tanggal = ? 
                ORDER BY datetime
            """, (date_str,))
            
            weather_rows = cursor.fetchall()
            conn.close()
            
            # Aggregate pollutant data: mean, max, median
            pm25_values = [float(row['pm2_5']) for row in polutan_rows if row['pm2_5'] is not None]
            o3_values = [float(row['ozone']) for row in polutan_rows if row['ozone'] is not None]
            co_values = [float(row['carbon_monoxide']) for row in polutan_rows if row['carbon_monoxide'] is not None]
            
            if not pm25_values or not o3_values or not co_values:
                return None
            
            agg_data = {
                'tanggal': target_date,
                'pm2_5 (μg/m³)_mean': np.mean(pm25_values),
                'pm2_5 (μg/m³)_max': np.max(pm25_values),
                'pm2_5 (μg/m³)_median': np.median(pm25_values),
                'ozone (μg/m³)_mean': np.mean(o3_values),
                'ozone (μg/m³)_max': np.max(o3_values),
                'ozone (μg/m³)_median': np.median(o3_values),
                'carbon_monoxide (μg/m³)_mean': np.mean(co_values),
                'carbon_monoxide (μg/m³)_max': np.max(co_values),
                'carbon_monoxide (μg/m³)_median': np.median(co_values),
            }
            
            # Aggregate weather data
            if weather_rows:
                temp_values = [float(row['temp']) for row in weather_rows if row['temp'] is not None]
                humidity_values = [float(row['humidity']) for row in weather_rows if row['humidity'] is not None]
                windspeed_values = [float(row['windspeed']) for row in weather_rows if row['windspeed'] is not None]
                winddir_values = [float(row['winddir']) for row in weather_rows if row['winddir'] is not None]
                pressure_values = [float(row['sealevelpressure']) for row in weather_rows if row['sealevelpressure'] is not None]
                cloudcover_values = [float(row['cloudcover']) for row in weather_rows if row['cloudcover'] is not None]
                visibility_values = [float(row['visibility']) for row in weather_rows if row['visibility'] is not None]
                solarrad_values = [float(row['solarradiation']) for row in weather_rows if row['solarradiation'] is not None]
                
                if temp_values:
                    agg_data['tempmin'] = np.min(temp_values)
                    agg_data['tempmax'] = np.max(temp_values)
                    agg_data['temp'] = np.mean(temp_values)
                if humidity_values:
                    agg_data['humidity'] = np.mean(humidity_values)
                if windspeed_values:
                    agg_data['windspeed'] = np.mean(windspeed_values)
                if winddir_values:
                    agg_data['winddir'] = np.mean(winddir_values)
                if pressure_values:
                    agg_data['sealevelpressure'] = np.mean(pressure_values)
                if cloudcover_values:
                    agg_data['cloudcover'] = np.mean(cloudcover_values)
                if visibility_values:
                    agg_data['visibility'] = np.mean(visibility_values)
                if solarrad_values:
                    agg_data['solarradiation'] = np.mean(solarrad_values)
            
            # Add temporal features
            agg_data['hari'] = target_date.day
            agg_data['bulan'] = target_date.month
            agg_data['is_weekend'] = 1 if target_date.weekday() >= 5 else 0
            
            # Normalize weather features to [0, 1] range to match training data
            # Use typical min/max ranges for Indonesian tropical weather
            normalization_ranges = {
                'tempmax': (15, 35),      # Typical max temp in Celsius
                'tempmin': (15, 30),      # Typical min temp
                'temp': (15, 35),         # Typical mean temp
                'humidity': (40, 95),     # Typical humidity percentage
                'windspeed': (0, 30),     # Typical wind speed km/h
                'winddir': (0, 360),      # Wind direction degrees
                'sealevelpressure': (1010, 1015),  # hPa
                'cloudcover': (0, 100),   # Cloud cover percentage
                'visibility': (0, 50),    # km
                'solarradiation': (0, 800),  # W/m2
            }
            
            for feature, (min_val, max_val) in normalization_ranges.items():
                if feature in agg_data and agg_data[feature] is not None:
                    # Normalize to [0, 1]
                    raw_val = agg_data[feature]
                    if max_val > min_val:
                        normalized = (raw_val - min_val) / (max_val - min_val)
                        # Clip to [0, 1] range
                        agg_data[feature] = np.clip(normalized, 0.0, 1.0)
                    else:
                        agg_data[feature] = 0.5  # fallback to middle value
            
            return agg_data
            
        except Exception as e:
            print(f"⚠️ Error fetching from SQLite: {e}")
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