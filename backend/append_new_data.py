#!/usr/bin/env python3
"""
Append data baru dari database (Nov 2025 - Jan 2026) ke master dataset
"""
import pandas as pd
import sqlite3
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Constants
DATASET_PATH = Path('/Users/user/Desktop/SKRIPSI/HASIL/Air Quality/backend/data/dataset_preprocessed/dataset_preprocessed.csv')
DB_PATH = Path('/Users/user/Desktop/SKRIPSI/HASIL/Air Quality/backend/data/datacrawler.db')

def append_new_data():
    """
    Append data dari database ke master dataset
    """
    
    print("="*80)
    print("🔗 APPEND NEW DATA FROM DATABASE")
    print("="*80)
    
    # 1. Load current dataset
    print("\n📂 Loading current dataset...")
    df_master = pd.read_csv(DATASET_PATH)
    df_master['tanggal'] = pd.to_datetime(df_master['tanggal'])
    
    print(f"  ✅ Current: {len(df_master)} rows")
    print(f"  📅 Range: {df_master['tanggal'].min()} to {df_master['tanggal'].max()}")
    
    last_date = df_master['tanggal'].max()
    last_date_str = last_date.strftime('%Y-%m-%d')
    
    # 2. Load data dari database
    print("\n📊 Loading data from database...")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Get unique dates > last_date
    query = f"""
    SELECT DISTINCT tanggal FROM polutan 
    WHERE DATE(tanggal) > DATE('{last_date_str}')
    ORDER BY tanggal
    """
    
    df_dates = pd.read_sql_query(query, conn)
    unique_dates = pd.to_datetime(df_dates['tanggal']).dt.date.unique()
    unique_dates = sorted(unique_dates)
    
    print(f"  ✅ New dates in database: {len(unique_dates)}")
    if unique_dates:
        print(f"  📅 From {unique_dates[0]} to {unique_dates[-1]}")
    
    # 3. Aggregate daily from database
    print("\n🔄 Aggregating new data...")
    
    new_rows = []
    
    for date_val in unique_dates:
        date_str = str(date_val)
        
        # Polutan
        polutan_query = """
        SELECT pm2_5, carbon_monoxide, ozone
        FROM polutan
        WHERE tanggal LIKE ?
        """
        
        df_polut = pd.read_sql_query(polutan_query, conn, params=(f"{date_str}%",))
        
        # Cuaca
        cuaca_query = """
        SELECT temp, humidity, windspeed, 
               winddir, sealevelpressure, cloudcover, visibility, solarradiation
        FROM cuaca
        WHERE tanggal = ?
        """
        
        df_cuaca = pd.read_sql_query(cuaca_query, conn, params=(date_str,))
        
        if len(df_polut) > 0 and len(df_cuaca) > 0:
            # Aggregate polutan
            row = {
                'tanggal': date_str,
                'pm2_5 (μg/m³)_mean': df_polut['pm2_5'].mean(),
                'pm2_5 (μg/m³)_max': df_polut['pm2_5'].max(),
                'pm2_5 (μg/m³)_median': df_polut['pm2_5'].median(),
                'carbon_monoxide (μg/m³)_mean': df_polut['carbon_monoxide'].mean(),
                'carbon_monoxide (μg/m³)_max': df_polut['carbon_monoxide'].max(),
                'carbon_monoxide (μg/m³)_median': df_polut['carbon_monoxide'].median(),
                'ozone (μg/m³)_mean': df_polut['ozone'].mean(),
                'ozone (μg/m³)_max': df_polut['ozone'].max(),
                'ozone (μg/m³)_median': df_polut['ozone'].median(),
            }
            
            # Add cuaca data (only available columns)
            for col in ['temp', 'humidity', 'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation']:
                if col in df_cuaca.columns:
                    row[col] = df_cuaca[col].values[0]
            
            # Handle missing tempmax/tempmin (database doesn't have them, use temp as fallback)
            if 'temp' in row:
                row['tempmax'] = row.get('tempmax', row['temp'])
                row['tempmin'] = row.get('tempmin', row['temp'])
            
            new_rows.append(row)
    
    conn.close()
    
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_new['tanggal'] = pd.to_datetime(df_new['tanggal'])
        
        print(f"  ✅ Aggregated {len(df_new)} new days")
    else:
        print(f"  ⚠️ No new data to append")
        return df_master
    
    # 4. Append & remove duplicates
    print("\n🔗 Merging...")
    
    df_combined = pd.concat([df_master, df_new], ignore_index=True)
    df_combined = df_combined.sort_values('tanggal').reset_index(drop=True)
    
    # Remove duplicates (keep last)
    before_dedup = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['tanggal'], keep='last')
    after_dedup = len(df_combined)
    
    if before_dedup > after_dedup:
        print(f"  🔄 Removed {before_dedup - after_dedup} duplicate dates")
    
    print(f"  ✅ Combined: {len(df_combined)} rows")
    print(f"  📅 Range: {df_combined['tanggal'].min()} to {df_combined['tanggal'].max()}")
    
    # 5. Save
    print("\n💾 Saving...")
    df_combined.to_csv(DATASET_PATH, index=False)
    print(f"  ✅ Saved: {DATASET_PATH}")
    
    # 6. Summary
    print("\n" + "="*80)
    print("✅ APPENDED!")
    print("="*80)
    print(f"  Previous: {len(df_master)} rows")
    print(f"  New: {len(df_new)} rows")
    print(f"  Total: {len(df_combined)} rows")
    print(f"  Date range: {df_combined['tanggal'].min()} to {df_combined['tanggal'].max()}")
    print(f"  NULL values: {df_combined.isna().sum().sum()}")
    
    print(f"\n📊 New data (from database):")
    print(df_combined[['tanggal', 'pm2_5 (μg/m³)_mean', 'temp', 'humidity']].tail(len(df_new)).to_string(index=False))
    
    return df_combined


if __name__ == "__main__":
    df = append_new_data()
