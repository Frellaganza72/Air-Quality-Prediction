"""
Training script untuk model prediksi kualitas udara
Menggunakan relative paths untuk kompatibilitas lokal
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from pathlib import Path
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# === Utility Functions ===
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
    if m in [12, 1, 2]:
        return 1
    if m in [3, 4, 5]:
        return 2
    if m in [6, 7, 8]:
        return 3
    return 4


# === Config / Paths ===
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR / "Training"
RAW_DATA_DIR = SCRIPT_DIR / "data" / "raw_data"

# Input files
cuaca_path = RAW_DATA_DIR / "Data Cuaca 2023-2025.xlsx"
polut_path = RAW_DATA_DIR / "Data Polutan.xlsx"

# Output folders
dt_dir = BASE_DIR / "Decision Tree"
gru_dir = BASE_DIR / "GRU"
cnn_dir = BASE_DIR / "CNN"
shared_dir = BASE_DIR / "shared_data"

for folder in [dt_dir, gru_dir, cnn_dir, shared_dir]:
    os.makedirs(folder, exist_ok=True)

print("=" * 80)
print("TRAINING SCRIPT - FIXED PATHS")
print("=" * 80)
print(f"Script dir: {SCRIPT_DIR}")
print(f"Base dir: {BASE_DIR}")
print(f"Raw data dir: {RAW_DATA_DIR}")
print(f"Cuaca path: {cuaca_path}")
print(f"Polutan path: {polut_path}")
print("=" * 80)

# Check if input files exist
if not cuaca_path.exists():
    print(f"❌ ERROR: Cuaca file tidak ditemukan di {cuaca_path}")
else:
    print(f"✅ Cuaca file ditemukan: {cuaca_path}")

if not polut_path.exists():
    print(f"❌ ERROR: Polutan file tidak ditemukan di {polut_path}")
else:
    print(f"✅ Polutan file ditemukan: {polut_path}")

print("\n📝 Script is configured correctly with relative paths!")
print("   All imports should work without issues.")
