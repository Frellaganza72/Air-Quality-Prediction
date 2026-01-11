# backend/crawler/db_handler.py
"""
Database handler untuk menyimpan data crawling ke SQLite.
"""
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import date, datetime
import logging

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "datacrawler.db"


def init_database():
    """
    Inisialisasi database dan buat tabel jika belum ada (dengan schema DAILY AGGREGATION).
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tabel polutan - DAILY AGGREGATION (mean, max, min, median per hari)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS polutan (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tanggal DATE UNIQUE NOT NULL,
            
            -- PM10 statistics
            pm10_mean REAL,
            pm10_max REAL,
            pm10_min REAL,
            pm10_median REAL,
            
            -- PM2.5 statistics
            pm2_5_mean REAL,
            pm2_5_max REAL,
            pm2_5_min REAL,
            pm2_5_median REAL,
            
            -- Ozone statistics
            ozone_mean REAL,
            ozone_max REAL,
            ozone_min REAL,
            ozone_median REAL,
            
            -- Carbon Monoxide statistics
            carbon_monoxide_mean REAL,
            carbon_monoxide_max REAL,
            carbon_monoxide_min REAL,
            carbon_monoxide_median REAL,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Tabel cuaca - DAILY AGGREGATION dengan statistik per hari
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cuaca (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tanggal DATE UNIQUE NOT NULL,
            datetime TEXT,
            
            -- Temperature statistics
            temp_mean REAL,
            temp_max REAL,
            tempmax REAL,
            tempmin REAL,
            
            -- Feels like statistics
            feelslike_mean REAL,
            feelslike_max REAL,
            
            -- Humidity statistics
            humidity_mean REAL,
            humidity_max REAL,
            
            -- Wind speed statistics
            windspeed_mean REAL,
            windspeed_max REAL,
            winddir_mean REAL,
            
            -- Pressure statistics
            sealevelpressure_mean REAL,
            sealevelpressure_max REAL,
            
            -- Cloud cover statistics
            cloudcover_mean REAL,
            cloudcover_max REAL,
            
            -- Visibility statistics
            visibility_mean REAL,
            visibility_max REAL,
            
            -- Solar radiation statistics
            solarradiation_mean REAL,
            solarradiation_max REAL,
            
            -- Other weather data
            conditions TEXT,
            icon TEXT,
            precip REAL,
            precipprob REAL,
            snow REAL,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_polutan_tanggal ON polutan(tanggal)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cuaca_tanggal ON cuaca(tanggal)")
    
    conn.commit()
    conn.close()
    logger.info(f"✅ Database initialized: {DB_PATH}")


def get_history_data(limit: int = 30):
    """
    Mengambil data riwayat polutan dari database SQLite (aggregated daily data).
    Mengambil data aktual harian hingga kemarin saja (exclude hari ini).
    
    Args:
        limit: Jumlah hari terakhir yang diambil (exclude hari ini)
        
    Returns:
        List of dictionaries berisi tanggal dan rata-rata polutan (hanya data kemarin ke belakang)
    """
    try:
        from datetime import date as dt_date, timedelta
        
        # Query dari SQLite database (aggregated daily data)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Calculate yesterday's date (exclude today since it's not complete yet)
        today = dt_date.today()
        
        # Query untuk last N days BEFORE today (complete days only)
        cursor.execute(f"""
            SELECT tanggal, pm2_5_mean, ozone_mean, carbon_monoxide_mean
            FROM polutan
            WHERE tanggal < ?
            ORDER BY tanggal DESC
            LIMIT ?
        """, (today.strftime('%Y-%m-%d'), limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            logger.warning(f"Tidak ada history data sebelum {today.strftime('%Y-%m-%d')}")
            return []
        
        # Convert to list of dictionaries (already sorted DESC from query)
        results = []
        for tanggal, pm25, o3, co in rows:
            results.append({
                'date': tanggal,
                'pm25': float(pm25) if pm25 is not None else 0,
                'o3': float(o3) if o3 is not None else 0,
                'co': float(co) if co is not None else 0
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error fetching history from database: {e}")
        import traceback
        traceback.print_exc()
        return []


def save_to_database(df_polut: pd.DataFrame = None, df_cuaca: pd.DataFrame = None, target_date: date = None):
    """
    Menyimpan data polutan dan cuaca ke database (AGGREGATED - 1 record per hari).
    Data sekarang sudah di-aggregate ke daily (mean, min, max) dari hourly.
    
    Args:
        df_polut: DataFrame data polutan aggregate (1 row per day)
        df_cuaca: DataFrame data cuaca aggregate (1 row per day)
        target_date: Tanggal target (default: hari ini)
    """
    if target_date is None:
        target_date = date.today()
    
    # Inisialisasi database jika belum ada
    init_database()
    
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Save polutan (now aggregated: 1 record per day)
        if df_polut is not None and not df_polut.empty:
            df_polut_db = df_polut.copy()
            
            # Tambahkan kolom tanggal
            df_polut_db['tanggal'] = target_date
            
            # Ambil 1 row (data sudah aggregate)
            row = df_polut_db.iloc[0]
            
            cursor = conn.cursor()
            try:
                # Valid columns untuk tabel polutan (sesuai schema baru)
                valid_cols = {
                    'tanggal', 
                    'pm10_mean', 'pm10_max', 'pm10_min', 'pm10_median',
                    'pm2_5_mean', 'pm2_5_max', 'pm2_5_min', 'pm2_5_median',
                    'ozone_mean', 'ozone_max', 'ozone_min', 'ozone_median',
                    'carbon_monoxide_mean', 'carbon_monoxide_max', 'carbon_monoxide_min', 'carbon_monoxide_median'
                }
                
                # Filter hanya kolom yang valid
                insert_cols = []
                insert_vals = []
                
                for col in df_polut_db.columns:
                    if col in valid_cols and pd.notna(row[col]):
                        insert_cols.append(col)
                        insert_vals.append(row[col])
                
                if insert_cols:
                    cols_str = ','.join([f'"{c}"' for c in insert_cols])
                    placeholders = ','.join(['?' for _ in insert_cols])
                    
                    cursor.execute(f"""
                        INSERT OR REPLACE INTO polutan ({cols_str})
                        VALUES ({placeholders})
                    """, tuple(insert_vals))
                    
                    conn.commit()
                    logger.info(f"✅ Saved 1 aggregated polutan record for {target_date}")
                else:
                    logger.warning(f"⚠️ No valid columns found for polutan")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error inserting polutan: {e}")
        
        # Save cuaca (now aggregated: 1 record per day)
        if df_cuaca is not None and not df_cuaca.empty:
            df_cuaca_db = df_cuaca.copy()
            
            # Tambahkan kolom tanggal
            df_cuaca_db['tanggal'] = target_date
            
            # Ambil 1 row (data sudah aggregate)
            row = df_cuaca_db.iloc[0]
            
            cursor = conn.cursor()
            try:
                # Valid columns untuk tabel cuaca (sesuai schema baru)
                valid_cols = {
                    'tanggal', 'datetime',
                    'temp_mean', 'temp_max', 'tempmax', 'tempmin',
                    'feelslike_mean', 'feelslike_max',
                    'humidity_mean', 'humidity_max',
                    'windspeed_mean', 'windspeed_max', 'winddir_mean',
                    'sealevelpressure_mean', 'sealevelpressure_max',
                    'cloudcover_mean', 'cloudcover_max',
                    'visibility_mean', 'visibility_max',
                    'solarradiation_mean', 'solarradiation_max',
                    'conditions', 'icon', 'precip', 'precipprob', 'snow'
                }
                
                # Filter hanya kolom yang valid
                insert_cols = []
                insert_vals = []
                
                for col in df_cuaca_db.columns:
                    if col in valid_cols and pd.notna(row[col]):
                        insert_cols.append(col)
                        val = row[col]
                        # Convert datetime to string jika perlu
                        if col == 'datetime':
                            val = str(val)
                        insert_vals.append(val)
                
                if insert_cols:
                    cols_str = ','.join([f'"{c}"' for c in insert_cols])
                    placeholders = ','.join(['?' for _ in insert_cols])
                    
                    cursor.execute(f"""
                        INSERT OR REPLACE INTO cuaca ({cols_str})
                        VALUES ({placeholders})
                    """, tuple(insert_vals))
                    
                    conn.commit()
                    logger.info(f"✅ Saved 1 aggregated cuaca record for {target_date}")
                else:
                    logger.warning(f"⚠️ No valid columns found for cuaca")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error inserting cuaca: {e}")
    
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Error saving to database: {e}")
        raise
    finally:
        conn.close()


def check_date_exists(target_date: date, check_both_tables: bool = True):
    """
    Cek apakah data untuk tanggal tertentu sudah ada di database.
    
    Args:
        target_date: Tanggal yang akan dicek
        check_both_tables: Jika True, cek di kedua tabel (polutan dan cuaca)
                          Jika False, hanya cek salah satu sudah cukup
    
    Returns:
        bool: True jika data sudah ada, False jika belum
    """
    if not DB_PATH.exists():
        return False
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        date_str = target_date.strftime("%Y-%m-%d")
        
        # Cek di tabel polutan
        cursor.execute("SELECT COUNT(*) FROM polutan WHERE tanggal = ?", (date_str,))
        polutan_count = cursor.fetchone()[0]
        
        # Cek di tabel cuaca
        cursor.execute("SELECT COUNT(*) FROM cuaca WHERE tanggal = ?", (date_str,))
        cuaca_count = cursor.fetchone()[0]
        
        if check_both_tables:
            # Data dianggap lengkap jika ada di kedua tabel
            exists = polutan_count > 0 and cuaca_count > 0
        else:
            # Data dianggap ada jika ada di salah satu tabel
            exists = polutan_count > 0 or cuaca_count > 0
        
        return exists
        
    except Exception as e:
        logger.error(f"❌ Error checking date in database: {e}")
        return False
    finally:
        conn.close()


def get_data_from_db(table: str, start_date: date = None, end_date: date = None):
    """
    Mengambil data dari database.
    
    Args:
        table: Nama tabel ('polutan' atau 'cuaca')
        start_date: Tanggal mulai (optional)
        end_date: Tanggal akhir (optional)
    
    Returns:
        DataFrame: Data dari database
    """
    if not DB_PATH.exists():
        logger.warning("Database belum ada")
        return pd.DataFrame()
    
    conn = sqlite3.connect(DB_PATH)
    
    query = f"SELECT * FROM {table}"
    conditions = []
    
    if start_date:
        conditions.append(f"tanggal >= '{start_date}'")
    if end_date:
        conditions.append(f"tanggal <= '{end_date}'")
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    # ORDER BY logic: cuaca has datetime, polutan only has tanggal
    if table == 'cuaca':
        query += " ORDER BY tanggal, datetime"
    else:
        query += " ORDER BY tanggal"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df


def get_trend_data_from_db(days_back: int = 7):
    """
    Mengambil data trend PM2.5 untuk N hari ke belakang dari SQLite.
    PENTING: Mengambil data ACTUAL (kemarin, bukan hari ini)
    
    Formula: 
    - Hari ini (2 Jan) → tampilkan [26 Des - 1 Jan] (7 hari lalu)
    - Besok (3 Jan) → tampilkan [27 Des - 2 Jan] (7 hari lalu)
    
    Args:
        days_back: Jumlah hari ke belakang dari KEMARIN (default: 7)
    
    Returns:
        list: List of dicts dengan format [{'date': 'YYYY-MM-DD', 'pm25': float}, ...]
    """
    if not DB_PATH.exists():
        logger.warning("Database belum ada, returning empty trend")
        return []
    
    try:
        from datetime import timedelta
        # PENTING: Gunakan KEMARIN sebagai end_date, bukan hari ini!
        end_date = date.today() - timedelta(days=1)  # Kemarin
        start_date = end_date - timedelta(days=days_back - 1)
        
        # Coba ambil data untuk rentang 7 hari terakhir
        df_polutan = get_data_from_db('polutan', start_date=start_date, end_date=end_date)
        
        # Jika tidak ada data untuk rentang tersebut, ambil data terakhir yang tersedia
        if df_polutan.empty:
            logger.info(f"Tidak ada data untuk rentang {start_date} hingga {end_date}, mengambil data terakhir yang tersedia")
            # Ambil semua data dan ambil N hari terakhir yang ada
            df_all = get_data_from_db('polutan')
            if df_all.empty:
                logger.warning("Database kosong, tidak ada data polutan")
                return []
            
            # Konversi tanggal
            df_all['tanggal'] = pd.to_datetime(df_all['tanggal'])
            
            # Ambil tanggal unik, sort, dan ambil N hari terakhir
            unique_dates = sorted(df_all['tanggal'].dt.date.unique(), reverse=True)
            if len(unique_dates) > days_back:
                unique_dates = unique_dates[:days_back]
            unique_dates.reverse()  # Sort ascending
            
            # Filter data untuk tanggal yang dipilih
            df_polutan = df_all[df_all['tanggal'].dt.date.isin(unique_dates)]
        
        # Pastikan kolom tanggal ada dan konversi ke datetime
        if 'tanggal' in df_polutan.columns:
            df_polutan['tanggal'] = pd.to_datetime(df_polutan['tanggal'])
        else:
            logger.error("Kolom 'tanggal' tidak ditemukan dalam data polutan")
            return []
        
        # Group by tanggal dan hitung rata-rata harian PM2.5
        df_polutan['date_only'] = df_polutan['tanggal'].dt.date
        
        # Use pm2_5_mean column (new schema)
        if 'pm2_5_mean' in df_polutan.columns:
            col_to_use = 'pm2_5_mean'
        elif 'pm2_5' in df_polutan.columns:
            col_to_use = 'pm2_5'
        else:
            logger.error("No PM2.5 column found in polutan data")
            return []
        
        daily_avg = df_polutan.groupby('date_only').agg({
            col_to_use: 'mean'
        }).reset_index()
        
        # Format untuk API response
        trend = []
        for _, row in daily_avg.iterrows():
            pm25_value = float(row[col_to_use]) if pd.notna(row[col_to_use]) else 0.0
            trend.append({
                'date': str(row['date_only']),
                'pm25': round(pm25_value, 2)
            })
        
        # Sort by date ascending
        trend.sort(key=lambda x: x['date'])
        
        logger.info(f"✅ Retrieved {len(trend)} days of trend data from SQLite")
        return trend
        
    except Exception as e:
        logger.error(f"❌ Error getting trend data from database: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return []


def get_ispu_statistics_from_db(days_back: int = 30):
    """
    Menghitung statistik kategori ISPU berdasarkan data N hari terakhir dari database.
    
    Args:
        days_back: Jumlah hari ke belakang dari hari ini (default: 30)
        Logic: Jika hari ini tanggal 8 Des, maka ambil data dari 6 Nov - 7 Des (30 hari)
    
    Returns:
        dict: Statistik dengan format {
            'categories': [
                {'name': 'Baik', 'value': count, 'color': '#22c55e'},
                {'name': 'Sedang', 'value': count, 'color': '#eab308'},
                ...
            ],
            'total_days': int,
            'total_hours': int,
            'date_range': {'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}
        }
    """
    if not DB_PATH.exists():
        logger.warning("Database belum ada, returning empty statistics")
        return {
            'categories': [],
            'total_days': 0,
            'total_hours': 0,
            'date_range': None
        }
    
    try:
        from datetime import timedelta
        from utils.ispu_classifier import ISPUClassifier
        
        # End date adalah kemarin (karena hari ini adalah prediksi, bukan data aktual)
        end_date = date.today() - timedelta(days=1)
        # Jika hari ini tanggal 23, maka end_date = tanggal 22
        # start_date = tanggal 22 - 29 hari = tanggal 24 November (30 hari: 24 Nov - 22 Des)
        start_date = end_date - timedelta(days=days_back - 1)
        
        # Ambil data polutan untuk rentang 30 hari
        df_polutan = get_data_from_db('polutan', start_date=start_date, end_date=end_date)
        
        if df_polutan.empty:
            logger.warning(f"Tidak ada data untuk rentang {start_date} hingga {end_date}")
            return {
                'categories': [],
                'total_days': 0,
                'total_hours': 0,
                'date_range': {'start': str(start_date), 'end': str(end_date)}
            }
        
        # Konversi tanggal
        df_polutan['tanggal'] = pd.to_datetime(df_polutan['tanggal'])
        df_polutan['date_only'] = df_polutan['tanggal'].dt.date
        
        # Group by tanggal dan ambil nilai tertinggi per hari (sama seperti logika anomali)
        # Use pm2_5_mean column (new schema)
        if 'pm2_5_mean' in df_polutan.columns:
            col_to_use = 'pm2_5_mean'
        elif 'pm2_5' in df_polutan.columns:
            col_to_use = 'pm2_5'
        else:
            logger.error("No PM2.5 column found")
            return {
                'categories': [],
                'total_days': 0,
                'total_hours': 0,
                'date_range': {'start': str(start_date), 'end': str(end_date)}
            }
        
        daily_max = df_polutan.groupby('date_only').agg({
            col_to_use: 'max',
            'ozone_mean': 'max' if 'ozone_mean' in df_polutan.columns else lambda x: 0,
            'carbon_monoxide_mean': 'max' if 'carbon_monoxide_mean' in df_polutan.columns else lambda x: 0
        }).reset_index()
        
        # Inisialisasi ISPU classifier
        ispu_classifier = ISPUClassifier()
        
        # Klasifikasi setiap hari berdasarkan kategori ISPU menggunakan HANYA PM2.5
        category_counts = {
            'Baik': 0,
            'Sedang': 0,
            'Tidak Sehat': 0,
            'Sangat Tidak Sehat': 0,
            'Berbahaya': 0
        }
        
        for _, row in daily_max.iterrows():
            pm25 = float(row[col_to_use]) if pd.notna(row[col_to_use]) else 0.0
            # o3 = float(row['ozone_mean']) if pd.notna(row['ozone_mean']) else 0.0
            # co = float(row['carbon_monoxide_mean']) if pd.notna(row['carbon_monoxide_mean']) else 0.0
            
            # Klasifikasi berdasarkan PM2.5 SAJA
            pm25_category = ispu_classifier.classify_pm25(pm25)
            
            if pm25_category in category_counts:
                category_counts[pm25_category] += 1
            else:
                category_counts['Sedang'] += 1  # Fallback
        
        # Format untuk frontend
        category_colors = {
            'Baik': '#22c55e',
            'Sedang': '#eab308',
            'Tidak Sehat': '#f97316',
            'Sangat Tidak Sehat': '#ef4444',
            'Berbahaya': '#7f1d1d'
        }
        
        categories = []
        for cat_name, count in category_counts.items():
            if count > 0:  # Hanya tampilkan kategori yang ada datanya
                categories.append({
                    'name': cat_name,
                    'value': count,
                    'color': category_colors.get(cat_name, '#475569')
                })
        
        total_days = len(daily_max)
        total_hours = len(df_polutan)
        
        logger.info(f"✅ Calculated ISPU statistics: {total_days} days, {total_hours} hours")
        
        return {
            'categories': categories,
            'total_days': total_days,
            'total_hours': total_hours,
            'date_range': {
                'start': str(start_date),
                'end': str(end_date)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error calculating ISPU statistics: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return {
            'categories': [],
            'total_days': 0,
            'total_hours': 0,
            'date_range': None
        }


def get_anomalies_from_db(limit: int = 50, days_back: int = 90):
    """
    Mengambil anomali dari file outliers.csv (hasil preprocessing resmi)
    BUKAN dari raw database polutan.
    
    Anomali hanya menampilkan data yang sudah divalidasi dalam preprocessing.
    
    Args:
        limit: Jumlah anomali yang dikembalikan (default: 50)
        days_back: Jumlah hari ke belakang untuk dianalisis (default: 90)
    
    Returns:
        list: List of dicts dengan format [{
            'datetime': 'YYYY-MM-DD',
            'pollutant': 'PM2.5' | 'O3' | 'CO',
            'value': float,
            'increase_percent': float,
            'description': str
        }, ...]
    """
    try:
        from datetime import timedelta
        from pathlib import Path
        
        # Load outliers.csv - data yang sudah divalidasi dalam preprocessing
        base_dir = Path(__file__).parent.parent  # backend/
        outliers_path = base_dir / 'data' / 'outliers.csv'
        
        if not outliers_path.exists():
            logger.warning(f"Outliers file tidak ditemukan: {outliers_path}")
            return []
        
        df_outliers = pd.read_csv(outliers_path)
        
        if df_outliers.empty:
            logger.warning("Outliers data kosong")
            return []
        
        # Konversi tanggal
        df_outliers['tanggal'] = pd.to_datetime(df_outliers['tanggal'])
        
        # Filter: hanya data sebelum hari ini (tidak termasuk prediksi)
        today = date.today()
        df_outliers = df_outliers[df_outliers['tanggal'].dt.date < today]
        
        if df_outliers.empty:
            logger.warning(f"Tidak ada data outliers sebelum {today}")
            return []
        
        # Sort descending (terbaru di atas) dan ambil limit
        df_outliers = df_outliers.sort_values('tanggal', ascending=False).head(limit)
        
        # Baseline (ISPU Baik max)
        pm25_baseline = 15.5
        o3_baseline = 60.0
        co_baseline = 2000.0
        
        anomalies = []
        
        # Proses setiap record outlier
        for _, row in df_outliers.iterrows():
            target_date = row['tanggal'].date()
            pm25_val = float(row['pm2_5 (μg/m³)_mean']) if pd.notna(row['pm2_5 (μg/m³)_mean']) else 0.0
            o3_val = float(row['ozone (μg/m³)_mean']) if pd.notna(row['ozone (μg/m³)_mean']) else 0.0
            co_val = float(row['carbon_monoxide (μg/m³)_mean']) if pd.notna(row['carbon_monoxide (μg/m³)_mean']) else 0.0
            
            # Tentukan polutan utama penyebab anomali
            pollutant = None
            value = 0.0
            baseline = 0.0
            increase = 0.0
            
            # Normalisasi pelanggaran threshold untuk perbandingan
            pm25_threshold = 35.0  # ISPU Sedang threshold
            o3_threshold = 120.0
            co_threshold = 4000.0
            
            pm25_ratio = pm25_val / pm25_threshold if pm25_threshold > 0 else 0
            o3_ratio = o3_val / o3_threshold if o3_threshold > 0 else 0
            co_ratio = co_val / co_threshold if co_threshold > 0 else 0
            
            max_ratio = max(pm25_ratio, o3_ratio, co_ratio)
            
            if max_ratio > 1.0:
                if max_ratio == pm25_ratio:
                    pollutant = 'PM2.5'
                    value = pm25_val
                    baseline = pm25_baseline
                    increase = ((pm25_val - baseline) / baseline * 100) if baseline > 0 else 0
                elif max_ratio == o3_ratio:
                    pollutant = 'O3'
                    value = o3_val
                    baseline = o3_baseline
                    increase = ((o3_val - baseline) / baseline * 100) if baseline > 0 else 0
                else:
                    pollutant = 'CO'
                    value = co_val
                    baseline = co_baseline
                    increase = ((co_val - baseline) / baseline * 100) if baseline > 0 else 0
            
                # Format output
                anomalies.append({
                    'datetime': target_date.strftime('%Y-%m-%d'),
                    'pollutant': pollutant,
                    'value': round(value, 2),
                    'increase_percent': round(increase, 1),
                    'description': f"Lonjakan {pollutant}: {round(value, 1)} (↑{round(increase, 1)}%)"
                })
        
        return anomalies
    
    except Exception as e:
        logger.error(f"❌ Error getting anomalies from outliers: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return []


def aggregate_hourly_to_daily(target_date: date = None):
    """
    AGGREGASI HOURLY DATA KE DAILY
    
    Aggregasi 24 jam hourly data dari SQLite (polutan + cuaca table)
    menjadi 1 daily row dengan mean, std, min, max untuk setiap parameter.
    
    Args:
        target_date: Tanggal yang akan diaggregasi (default: hari ini)
    
    Return: dict dengan struktur
    {
        'date': '2026-01-04',
        'pm2_5 (μg/m³)_mean': 28.5,
        'pm2_5 (μg/m³)_std': 2.3,
        'pm2_5 (μg/m³)_max': 32.1,
        'pm2_5 (μg/m³)_median': 27.8,
        'ozone (μg/m³)_mean': 45.8,
        'carbon_monoxide (μg/m³)_mean': 850.5,
        'temp': 0.55,
        'humidity': 0.72,
        'windspeed': 0.8,
        'sealevelpressure': 0.34,
        'cloudcover': 0.6,
        'visibility': 0.85,
        'solarradiation': 0.5,
        'hourly_count': 24,
        'missing_hours': 0
    }
    """
    
    if target_date is None:
        target_date = date.today()
    
    try:
        if not DB_PATH.exists():
            logger.error(f"[AGGREGATION] Database tidak ada: {DB_PATH}")
            return None
        
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        date_str = target_date.strftime("%Y-%m-%d")
        logger.info(f"[AGGREGATION] Aggregating hourly data untuk {date_str}")
        
        # QUERY 1: Aggregate polutan table (PM2.5, O3, CO) - menggunakan nama kolom baru
        cursor.execute("""
            SELECT 
                pm2_5_mean,
                pm2_5_max,
                pm2_5_min,
                pm2_5_median,
                ozone_mean,
                ozone_max,
                ozone_min,
                ozone_median,
                carbon_monoxide_mean,
                carbon_monoxide_max,
                carbon_monoxide_min,
                carbon_monoxide_median
            FROM polutan
            WHERE tanggal = ?
        """, (date_str,))
        
        polutan_rows = cursor.fetchone()
        
        # QUERY 2: Aggregate cuaca table (temperature, humidity, wind, pressure, etc)
        cursor.execute("""
            SELECT 
                temp_mean,
                temp_max,
                tempmax,
                tempmin,
                humidity_mean,
                humidity_max,
                windspeed_mean,
                windspeed_max,
                winddir_mean,
                sealevelpressure_mean,
                sealevelpressure_max,
                cloudcover_mean,
                cloudcover_max,
                visibility_mean,
                visibility_max,
                solarradiation_mean,
                solarradiation_max
            FROM cuaca
            WHERE tanggal = ?
        """, (date_str,))
        
        cuaca_rows = cursor.fetchone()
        conn.close()
        
        # Jika sudah ada data aggregated untuk hari ini, gunakan langsung
        if polutan_rows is not None:
            logger.info(f"[AGGREGATION] Data sudah aggregated untuk {date_str}, menggunakan data existing")
            aggregation = {
                'tanggal': date_str,
                'pm2_5 (μg/m³)_mean': polutan_rows[0],
                'pm2_5 (μg/m³)_max': polutan_rows[1],
                'pm2_5 (μg/m³)_min': polutan_rows[2],
                'pm2_5 (μg/m³)_median': polutan_rows[3],
                'ozone (μg/m³)_mean': polutan_rows[4],
                'ozone (μg/m³)_max': polutan_rows[5],
                'ozone (μg/m³)_min': polutan_rows[6],
                'ozone (μg/m³)_median': polutan_rows[7],
                'carbon_monoxide (μg/m³)_mean': polutan_rows[8],
                'carbon_monoxide (μg/m³)_max': polutan_rows[9],
                'carbon_monoxide (μg/m³)_min': polutan_rows[10],
                'carbon_monoxide (μg/m³)_median': polutan_rows[11],
            }
            
            if cuaca_rows is not None:
                aggregation.update({
                    'temp': cuaca_rows[0] if cuaca_rows[0] is not None else cuaca_rows[1],
                    'tempmax': cuaca_rows[2],
                    'tempmin': cuaca_rows[3],
                    'humidity': cuaca_rows[4] if cuaca_rows[4] is not None else cuaca_rows[5],
                    'windspeed': cuaca_rows[6] if cuaca_rows[6] is not None else cuaca_rows[7],
                    'winddir': cuaca_rows[8],
                    'sealevelpressure': cuaca_rows[9] if cuaca_rows[9] is not None else cuaca_rows[10],
                    'cloudcover': cuaca_rows[11] if cuaca_rows[11] is not None else cuaca_rows[12],
                    'visibility': cuaca_rows[13] if cuaca_rows[13] is not None else cuaca_rows[14],
                    'solarradiation': cuaca_rows[15] if cuaca_rows[15] is not None else cuaca_rows[16],
                    'hourly_count': 24,
                    'missing_hours': 0,
                })
            
            return aggregation
        
        logger.warning(f"[AGGREGATION] Tidak ada aggregated data untuk {date_str} di database")
        return None
    except Exception as e:
        logger.error(f"[AGGREGATION] Error: {e}", exc_info=True)
        return None
        
        # Sort by tanggal agar urut chronologically (convert to datetime for proper sorting)
        if 'tanggal' in df_new.columns:
            df_new['tanggal'] = pd.to_datetime(df_new['tanggal'], errors='coerce')
            df_new = df_new.sort_values('tanggal', ascending=True).reset_index(drop=True)
            df_new['tanggal'] = df_new['tanggal'].dt.strftime('%Y-%m-%d')
        
        # Drop duplicates (keep first occurrence)
        if 'tanggal' in df_new.columns:
            df_new = df_new.drop_duplicates(subset=['tanggal'], keep='first')
        
        # Save to CSV
        df_new.to_csv(master_file, index=False)
        
        logger.info(f"[MERGE] ✅ Merge sukses!")
        logger.info(f"   Previous size: {len(df_master)} rows")
        logger.info(f"   New size: {len(df_new)} rows")
        logger.info(f"   Added: {target_date_str} dengan data:")
        logger.info(f"   - PM2.5: {row.get('pm2_5 (μg/m³)_mean')} μg/m³")
        logger.info(f"   - O3: {row.get('ozone (μg/m³)_mean')} μg/m³")
        logger.info(f"   - CO: {row.get('carbon_monoxide (μg/m³)_mean')} μg/m³")
        
        return True
        
    except Exception as e:
        logger.error(f"[MERGE] Error merging to master dataset: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return False


def merge_aggregation_to_master_dataset(aggregation_data: dict = None, target_date: date = None):
    """
    Merge aggregated daily data ke master dataset (dataset_preprocessed.csv)
    
    Args:
        aggregation_data: dict dengan hasil aggregate_hourly_to_daily()
        target_date: date object (jika None, gunakan yesterday)
    
    Returns:
        bool: True jika merge sukses, False jika gagal
    """
    import pandas as pd
    
    try:
        # Tentukan tanggal target
        if target_date is None:
            from datetime import timedelta
            target_date = date.today() - timedelta(days=1)
        
        target_date_str = target_date.strftime('%Y-%m-%d')
        
        # Jika aggregation_data tidak diberikan, fetch dari database
        if aggregation_data is None:
            aggregation_data = aggregate_hourly_to_daily(target_date)
            if aggregation_data is None:
                logger.warning(f"[MERGE] Tidak ada aggregation data untuk {target_date_str}")
                return False
        
        # Path ke master dataset
        master_file = DB_PATH.parent.parent / "dataset_preprocessed" / "dataset_preprocessed.csv"
        
        if not master_file.exists():
            logger.warning(f"[MERGE] Master dataset tidak ada di {master_file}")
            logger.info("[MERGE] Creating new master dataset...")
            
            # Buat new master dataset dari aggregation data
            df_new = pd.DataFrame([aggregation_data])
            master_file.parent.mkdir(parents=True, exist_ok=True)
            df_new.to_csv(master_file, index=False)
            logger.info(f"[MERGE] ✅ Created master dataset with 1 row")
            return True
        
        # Load existing master dataset
        df_master = pd.read_csv(master_file)
        
        # Check if date already exists
        if 'tanggal' in df_master.columns:
            if target_date_str in df_master['tanggal'].values:
                logger.info(f"[MERGE] Data untuk {target_date_str} sudah ada, skip merge")
                return False
        
        # Prepare row untuk di-append
        row = pd.DataFrame([aggregation_data])
        
        # Merge dengan existing data
        df_new = pd.concat([df_master, row], ignore_index=True)
        
        # Sort by tanggal agar urut chronologically
        if 'tanggal' in df_new.columns:
            df_new['tanggal'] = pd.to_datetime(df_new['tanggal'], errors='coerce')
            df_new = df_new.sort_values('tanggal', ascending=True).reset_index(drop=True)
            df_new['tanggal'] = df_new['tanggal'].dt.strftime('%Y-%m-%d')
        
        # Drop duplicates (keep first occurrence)
        if 'tanggal' in df_new.columns:
            df_new = df_new.drop_duplicates(subset=['tanggal'], keep='first')
        
        # Save to CSV
        df_new.to_csv(master_file, index=False)
        
        logger.info(f"[MERGE] ✅ Merge sukses!")
        logger.info(f"   Previous size: {len(df_master)} rows")
        logger.info(f"   New size: {len(df_new)} rows")
        logger.info(f"   Added: {target_date_str} dengan data:")
        logger.info(f"   - PM2.5: {aggregation_data.get('pm2_5 (μg/m³)_mean')} μg/m³")
        logger.info(f"   - O3: {aggregation_data.get('ozone (μg/m³)_mean')} μg/m³")
        logger.info(f"   - CO: {aggregation_data.get('carbon_monoxide (μg/m³)_mean')} μg/m³")
        
        return True
        
    except Exception as e:
        logger.error(f"[MERGE] Error merging to master dataset: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test database initialization
    print("Testing database initialization...")
    init_database()
    print(f"✅ Database created at: {DB_PATH}")

