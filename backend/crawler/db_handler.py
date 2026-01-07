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
    Inisialisasi database dan buat tabel jika belum ada.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tabel polutan
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS polutan (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            waktu TIMESTAMP NOT NULL,
            pm10 REAL,
            pm2_5 REAL,
            ozone REAL,
            carbon_monoxide REAL,
            tanggal DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(waktu, tanggal)
        )
    """)
    
    # Tabel cuaca
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cuaca (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TIMESTAMP NOT NULL,
            name TEXT,
            temp REAL,
            feelslike REAL,
            dew REAL,
            humidity REAL,
            precip REAL,
            precipprob REAL,
            preciptype TEXT,
            snow REAL,
            snowdepth REAL,
            windgust REAL,
            windspeed REAL,
            winddir REAL,
            sealevelpressure REAL,
            cloudcover REAL,
            visibility REAL,
            solarradiation REAL,
            solarenergy REAL,
            uvindex REAL,
            severerisk REAL,
            conditions TEXT,
            icon TEXT,
            stations TEXT,
            tanggal DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(datetime, tanggal)
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_polutan_tanggal ON polutan(tanggal)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_polutan_waktu ON polutan(waktu)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cuaca_tanggal ON cuaca(tanggal)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cuaca_datetime ON cuaca(datetime)")
    
    conn.commit()
    conn.close()
    logger.info(f"✅ Database initialized: {DB_PATH}")


def get_history_data(limit: int = 30):
    """
    Mengambil data riwayat polutan dari master dataset (bukan dari SQLite yang outdated).
    Mengambil data aktual harian hingga kemarin saja.
    
    Args:
        limit: Jumlah hari terakhir yang diambil (exclude hari ini)
        
    Returns:
        List of dictionaries berisi tanggal dan rata-rata polutan (hanya data kemarin ke belakang)
    """
    try:
        # Gunakan master dataset alih-alih SQLite (lebih reliable dan selalu updated)
        preprocessed_path = DB_PATH.parent / 'dataset_preprocessed' / 'dataset_preprocessed.csv'
        
        if not preprocessed_path.exists():
            logger.warning(f"Master dataset not found at {preprocessed_path}")
            return []
        
        # Load master dataset
        df = pd.read_csv(preprocessed_path)
        
        if df.empty or 'tanggal' not in df.columns:
            logger.warning("Master dataset kosong atau tidak punya kolom 'tanggal'")
            return []
        
        # Convert tanggal ke datetime
        df['tanggal'] = pd.to_datetime(df['tanggal'])
        today = pd.to_datetime(date.today())
        
        # Filter: hanya data sebelum hari ini (exclude prediksi hari ini)
        df_history = df[df['tanggal'] < today].copy()
        
        if df_history.empty:
            logger.warning(f"Tidak ada history data sebelum {today.strftime('%Y-%m-%d')}")
            return []
        
        # Sort descending dan ambil limit hari
        df_history = df_history.sort_values('tanggal', ascending=False).head(limit)
        
        # Extract columns
        results = []
        for _, row in df_history.iterrows():
            # Handle columns yang mungkin punya nama berbeda
            pm25_val = row.get('pm2_5 (μg/m³)_mean') or row.get('pm25') or 0
            o3_val = row.get('ozone (μg/m³)_mean') or row.get('o3') or 0
            co_val = row.get('carbon_monoxide (μg/m³)_mean') or row.get('co') or 0
            
            results.append({
                'date': row['tanggal'].strftime('%Y-%m-%d'),
                'pm25': float(pm25_val),
                'o3': float(o3_val),
                'co': float(co_val)
            })
        
        # Return results (already sorted DESC from query)
        return results
        
    except Exception as e:
        logger.error(f"Error fetching history from master dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


def save_to_database(df_polut: pd.DataFrame = None, df_cuaca: pd.DataFrame = None, target_date: date = None):
    """
    Menyimpan data polutan dan cuaca ke database.
    
    Args:
        df_polut: DataFrame data polutan
        df_cuaca: DataFrame data cuaca
        target_date: Tanggal target (default: hari ini)
    """
    if target_date is None:
        target_date = date.today()
    
    # Inisialisasi database jika belum ada
    init_database()
    
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Save polutan
        if df_polut is not None and not df_polut.empty:
            # Rename kolom time menjadi waktu jika ada
            df_polut_db = df_polut.copy()
            if 'time' in df_polut_db.columns:
                df_polut_db = df_polut_db.rename(columns={'time': 'waktu'})
            
            # Tambahkan kolom tanggal
            df_polut_db['tanggal'] = target_date
            
            # Pastikan kolom waktu adalah timestamp, lalu konversi ke string
            if 'waktu' in df_polut_db.columns:
                df_polut_db['waktu'] = pd.to_datetime(df_polut_db['waktu'])
                # Konversi Timestamp ke string untuk SQLite
                df_polut_db['waktu'] = df_polut_db['waktu'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Pilih kolom yang sesuai dengan tabel
            cols_polut = ['waktu', 'pm10', 'pm2_5', 'ozone', 'carbon_monoxide', 'tanggal']
            df_polut_db = df_polut_db[[c for c in cols_polut if c in df_polut_db.columns]]
            
            # Insert dengan OR IGNORE untuk menghindari duplicate
            cursor = conn.cursor()
            for _, row in df_polut_db.iterrows():
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO polutan (waktu, pm10, pm2_5, ozone, carbon_monoxide, tanggal)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        str(row['waktu']) if 'waktu' in row and pd.notna(row['waktu']) else None,
                        float(row['pm10']) if 'pm10' in row and pd.notna(row['pm10']) else None,
                        float(row['pm2_5']) if 'pm2_5' in row and pd.notna(row['pm2_5']) else None,
                        float(row['ozone']) if 'ozone' in row and pd.notna(row['ozone']) else None,
                        float(row['carbon_monoxide']) if 'carbon_monoxide' in row and pd.notna(row['carbon_monoxide']) else None,
                        str(row['tanggal'])
                    ))
                except Exception as e:
                    logger.warning(f"⚠️ Error inserting row: {e}")
                    continue
            conn.commit()
            logger.info(f"✅ Saved {len(df_polut_db)} polutan records to database")
        
        # Save cuaca
        if df_cuaca is not None and not df_cuaca.empty:
            df_cuaca_db = df_cuaca.copy()
            
            # Tambahkan kolom tanggal
            df_cuaca_db['tanggal'] = target_date
            
            # Pastikan datetime adalah timestamp, lalu konversi ke string
            if 'datetime' in df_cuaca_db.columns:
                df_cuaca_db['datetime'] = pd.to_datetime(df_cuaca_db['datetime'])
                # Konversi Timestamp ke string untuk SQLite
                df_cuaca_db['datetime'] = df_cuaca_db['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Insert dengan OR IGNORE untuk menghindari duplicate
            cursor = conn.cursor()
            cuaca_cols = [col for col in df_cuaca_db.columns]
            
            for _, row in df_cuaca_db.iterrows():
                try:
                    # Konversi semua nilai ke format yang didukung SQLite
                    values = []
                    for col in cuaca_cols:
                        val = row[col]
                        if pd.isna(val):
                            values.append(None)
                        elif isinstance(val, (pd.Timestamp, datetime)):
                            values.append(val.strftime('%Y-%m-%d %H:%M:%S') if hasattr(val, 'strftime') else str(val))
                        elif isinstance(val, (int, float)):
                            values.append(float(val))
                        else:
                            values.append(str(val))
                    
                    cols_placeholders = ','.join(['?' for _ in cuaca_cols])
                    cols_names = ','.join([f'"{col}"' for col in cuaca_cols])  # Quote column names
                    
                    cursor.execute(f"""
                        INSERT OR IGNORE INTO cuaca ({cols_names})
                        VALUES ({cols_placeholders})
                    """, tuple(values))
                except Exception as e:
                    logger.warning(f"⚠️ Error inserting cuaca row: {e}")
                    continue
            conn.commit()
            logger.info(f"✅ Saved {len(df_cuaca_db)} cuaca records to database")
        
        conn.commit()
        
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
    
    query += " ORDER BY tanggal, datetime" if table == 'cuaca' else " ORDER BY tanggal, waktu"
    
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
        daily_avg = df_polutan.groupby('date_only').agg({
            'pm2_5': 'mean'
        }).reset_index()
        
        # Format untuk API response
        trend = []
        for _, row in daily_avg.iterrows():
            pm25_value = float(row['pm2_5']) if pd.notna(row['pm2_5']) else 0.0
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
        daily_max = df_polutan.groupby('date_only').agg({
            'pm2_5': 'max',
            'ozone': 'max',
            'carbon_monoxide': 'max'
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
            pm25 = float(row['pm2_5']) if pd.notna(row['pm2_5']) else 0.0
            # o3 = float(row['ozone']) if pd.notna(row['ozone']) else 0.0
            # co = float(row['carbon_monoxide']) if pd.notna(row['carbon_monoxide']) else 0.0
            
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
        
        # QUERY 1: Aggregate polutan table (PM2.5, O3, CO)
        cursor.execute("""
            SELECT 
                pm2_5,
                ozone,
                carbon_monoxide
            FROM polutan
            WHERE tanggal = ?
            ORDER BY waktu
        """, (date_str,))
        
        polutan_rows = cursor.fetchall()
        
        # QUERY 2: Aggregate cuaca table (temperature, humidity, wind, pressure, etc)
        cursor.execute("""
            SELECT 
                temp,
                humidity,
                windspeed,
                sealevelpressure,
                cloudcover,
                visibility,
                solarradiation
            FROM cuaca
            WHERE tanggal = ?
            ORDER BY datetime
        """, (date_str,))
        
        cuaca_rows = cursor.fetchall()
        conn.close()
        
        # Konversi ke DataFrame dengan column names
        polutan_columns = ['pm2_5', 'ozone', 'carbon_monoxide']
        cuaca_columns = ['temp', 'humidity', 'windspeed', 'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation']
        
        df_polutan = pd.DataFrame(polutan_rows, columns=polutan_columns) if polutan_rows else pd.DataFrame()
        df_cuaca = pd.DataFrame(cuaca_rows, columns=cuaca_columns) if cuaca_rows else pd.DataFrame()
        
        logger.info(f"[AGGREGATION] Polutan records: {len(df_polutan)}, Cuaca records: {len(df_cuaca)}")
        
        if df_polutan.empty and df_cuaca.empty:
            logger.warning(f"[AGGREGATION] Tidak ada data untuk {date_str}")
            return None
        
        # Initialize hasil agregasi
        aggregation = {
            'tanggal': date_str,
            'hourly_count': max(len(df_polutan), len(df_cuaca)),
            'missing_hours': 24 - max(len(df_polutan), len(df_cuaca)),
        }
        
        # ---- AGGREGASI POLUTAN ----
        # PM2.5
        if not df_polutan.empty and 'pm2_5' in df_polutan.columns:
            pm25_data = pd.to_numeric(df_polutan['pm2_5'], errors='coerce').dropna()
            if len(pm25_data) > 0:
                aggregation['pm2_5 (μg/m³)_mean'] = round(float(pm25_data.mean()), 2)
                aggregation['pm2_5 (μg/m³)_max'] = round(float(pm25_data.max()), 2)
                aggregation['pm2_5 (μg/m³)_median'] = round(float(pm25_data.median()), 2)
                logger.info(f"   PM2.5: mean={aggregation['pm2_5 (μg/m³)_mean']}, max={aggregation['pm2_5 (μg/m³)_max']}")
        
        # Ozone
        if not df_polutan.empty and 'ozone' in df_polutan.columns:
            o3_data = pd.to_numeric(df_polutan['ozone'], errors='coerce').dropna()
            if len(o3_data) > 0:
                aggregation['ozone (μg/m³)_mean'] = round(float(o3_data.mean()), 2)
                aggregation['ozone (μg/m³)_max'] = round(float(o3_data.max()), 2)
                aggregation['ozone (μg/m³)_median'] = round(float(o3_data.median()), 2)
                logger.info(f"   O3: mean={aggregation['ozone (μg/m³)_mean']}, max={aggregation['ozone (μg/m³)_max']}")
        
        # Carbon Monoxide
        if not df_polutan.empty and 'carbon_monoxide' in df_polutan.columns:
            co_data = pd.to_numeric(df_polutan['carbon_monoxide'], errors='coerce').dropna()
            if len(co_data) > 0:
                aggregation['carbon_monoxide (μg/m³)_mean'] = round(float(co_data.mean()), 2)
                aggregation['carbon_monoxide (μg/m³)_max'] = round(float(co_data.max()), 2)
                aggregation['carbon_monoxide (μg/m³)_median'] = round(float(co_data.median()), 2)
                logger.info(f"   CO: mean={aggregation['carbon_monoxide (μg/m³)_mean']}, max={aggregation['carbon_monoxide (μg/m³)_max']}")
        
        # ---- AGGREGASI CUACA ----
        # Temperature (normalized 0-1)
        if not df_cuaca.empty and 'temp' in df_cuaca.columns:
            temp_data = pd.to_numeric(df_cuaca['temp'], errors='coerce').dropna()
            if len(temp_data) > 0:
                temp_avg = float(temp_data.mean())
                aggregation['temp'] = round(temp_avg / 100.0, 3)  # normalize to 0-1 range
                aggregation['tempmax'] = round(float(temp_data.max()) / 100.0, 3)
                aggregation['tempmin'] = round(float(temp_data.min()) / 100.0, 3)
                logger.info(f"   Temp: avg={temp_avg}°C (normalized={aggregation['temp']})")
        
        # Humidity (0-1 range)
        if not df_cuaca.empty and 'humidity' in df_cuaca.columns:
            humidity_data = pd.to_numeric(df_cuaca['humidity'], errors='coerce').dropna()
            if len(humidity_data) > 0:
                humidity_avg = float(humidity_data.mean())
                aggregation['humidity'] = round(humidity_avg / 100.0, 3)
                logger.info(f"   Humidity: avg={humidity_avg}% (normalized={aggregation['humidity']})")
        
        # Wind Speed (0-1 range)
        if not df_cuaca.empty and 'windspeed' in df_cuaca.columns:
            windspeed_data = pd.to_numeric(df_cuaca['windspeed'], errors='coerce').dropna()
            if len(windspeed_data) > 0:
                windspeed_avg = float(windspeed_data.mean())
                aggregation['windspeed'] = round(windspeed_avg / 50.0, 3)  # normalize by max typical
                aggregation['winddir'] = round(float(windspeed_data.iloc[0] if len(windspeed_data) > 0 else 0) / 360.0, 3)
                logger.info(f"   Wind Speed: avg={windspeed_avg} km/h (normalized={aggregation['windspeed']})")
        
        # Sea Level Pressure (0-1 range, typical 1000-1030 hPa)
        if not df_cuaca.empty and 'sealevelpressure' in df_cuaca.columns:
            pressure_data = pd.to_numeric(df_cuaca['sealevelpressure'], errors='coerce').dropna()
            if len(pressure_data) > 0:
                pressure_avg = float(pressure_data.mean())
                aggregation['sealevelpressure'] = round((pressure_avg - 1000.0) / 100.0, 3)  # normalize
                logger.info(f"   Pressure: avg={pressure_avg} hPa (normalized={aggregation['sealevelpressure']})")
        
        # Cloud Cover (0-1 range)
        if not df_cuaca.empty and 'cloudcover' in df_cuaca.columns:
            cloudcover_data = pd.to_numeric(df_cuaca['cloudcover'], errors='coerce').dropna()
            if len(cloudcover_data) > 0:
                cloudcover_avg = float(cloudcover_data.mean())
                aggregation['cloudcover'] = round(cloudcover_avg / 100.0, 3)
                logger.info(f"   Cloud Cover: avg={cloudcover_avg}% (normalized={aggregation['cloudcover']})")
        
        # Visibility (0-1 range, typical 0-20km)
        if not df_cuaca.empty and 'visibility' in df_cuaca.columns:
            visibility_data = pd.to_numeric(df_cuaca['visibility'], errors='coerce').dropna()
            if len(visibility_data) > 0:
                visibility_avg = float(visibility_data.mean())
                aggregation['visibility'] = round(visibility_avg / 20.0, 3)
                logger.info(f"   Visibility: avg={visibility_avg} km (normalized={aggregation['visibility']})")
        
        # Solar Radiation (0-1 range)
        if not df_cuaca.empty and 'solarradiation' in df_cuaca.columns:
            solar_data = pd.to_numeric(df_cuaca['solarradiation'], errors='coerce').dropna()
            if len(solar_data) > 0:
                solar_avg = float(solar_data.mean())
                aggregation['solarradiation'] = round(solar_avg / 1000.0, 3)  # normalize
                logger.info(f"   Solar Radiation: avg={solar_avg} W/m² (normalized={aggregation['solarradiation']})")
        
        logger.info("[AGGREGATION] ✅ Aggregation berhasil!")
        return aggregation
        
    except Exception as e:
        logger.error(f"[AGGREGATION] Error during aggregation: {e}", exc_info=True)
        return None


def merge_aggregation_to_master_dataset(aggregation):
    """
    MERGE HASIL AGREGASI KE MASTER DATASET
    
    Merge 1 baris daily data agregasi ke dataset_preprocessed.csv untuk continuous learning.
    Dataset ini akan digunakan untuk training setiap hari dengan data lama + data baru.
    
    Input: 
        aggregation dict dari aggregate_hourly_to_daily()
        
    Output: 
        Update/create dataset_preprocessed.csv dengan 1 baris baru
        Return: True jika sukses, False jika skip/gagal
    """
    
    try:
        import numpy as np
        
        # Path ke master dataset (preprocessed)
        master_file = Path(__file__).resolve().parents[1] / 'data' / 'dataset_preprocessed' / 'dataset_preprocessed.csv'
        master_file.parent.mkdir(parents=True, exist_ok=True)
        
        if aggregation is None:
            logger.warning("[MERGE] Aggregation data is None, skip merge")
            return False
        
        target_date_str = aggregation.get('tanggal')
        
        # Prepare row untuk dimasukkan ke master dataset
        row = {
            'tanggal': target_date_str,
            'pm2_5 (μg/m³)_mean': aggregation.get('pm2_5 (μg/m³)_mean'),
            'pm2_5 (μg/m³)_max': aggregation.get('pm2_5 (μg/m³)_max'),
            'pm2_5 (μg/m³)_median': aggregation.get('pm2_5 (μg/m³)_median'),
            'ozone (μg/m³)_mean': aggregation.get('ozone (μg/m³)_mean'),
            'ozone (μg/m³)_max': aggregation.get('ozone (μg/m³)_max'),
            'ozone (μg/m³)_median': aggregation.get('ozone (μg/m³)_median'),
            'carbon_monoxide (μg/m³)_mean': aggregation.get('carbon_monoxide (μg/m³)_mean'),
            'carbon_monoxide (μg/m³)_max': aggregation.get('carbon_monoxide (μg/m³)_max'),
            'carbon_monoxide (μg/m³)_median': aggregation.get('carbon_monoxide (μg/m³)_median'),
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
        
        # Load atau create master dataset
        if master_file.exists():
            df_master = pd.read_csv(master_file)
            logger.info(f"[MERGE] Loaded existing master dataset: {len(df_master)} rows")
        else:
            df_master = pd.DataFrame()
            logger.info(f"[MERGE] Master dataset tidak ada, akan create baru")
        
        # Check jika sudah ada untuk hari ini (prevent duplicate)
        if len(df_master) > 0 and 'tanggal' in df_master.columns:
            existing = df_master[df_master['tanggal'] == target_date_str]
            if len(existing) > 0:
                logger.warning(f"[MERGE] Data untuk {target_date_str} sudah ada, skip merge")
                return False
        
        # Append row baru
        df_new = pd.concat([df_master, pd.DataFrame([row])], ignore_index=True)
        
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


if __name__ == "__main__":
    # Test database initialization
    print("Testing database initialization...")
    init_database()
    print(f"✅ Database created at: {DB_PATH}")

