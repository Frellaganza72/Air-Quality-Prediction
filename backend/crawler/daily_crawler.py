# backend/crawler/daily_crawler.py
"""
Daily crawler untuk mengambil data polutan dan cuaca dari API eksternal.
Mengambil data 1 hari per crawling.
"""
import sys
from pathlib import Path
import requests
import pandas as pd
from datetime import date, timedelta
import os
import logging
import time
from dotenv import load_dotenv

# Fix path untuk import - tambahkan parent directory ke sys.path jika dijalankan langsung
if __name__ == "__main__":
    backend_dir = Path(__file__).resolve().parents[1]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

# Load environment variables
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def crawl_daily(target_date: date = None, save_to_db: bool = True, skip_if_exists: bool = True):
    """
    Crawl data polutan dan cuaca untuk 1 hari tertentu (default: hari ini).
    
    Args:
        target_date (date): Tanggal yang akan di-crawl (default: hari ini)
        save_to_db (bool): Simpan ke database SQLite (default: True)
        skip_if_exists (bool): Skip crawling jika data sudah ada di database (default: True)
    
    Returns:
        tuple: (df_polut, df_cuaca) jika berhasil, (None, None) jika gagal atau skip
    """
    try:
        latitude, longitude = -7.9797, 112.6304
        if target_date is None:
            target_date = date.today()
        
        today_str = target_date.strftime("%Y-%m-%d")
        
        # Cek apakah data sudah ada di database
        if skip_if_exists and save_to_db:
            try:
                from crawler.db_handler import check_date_exists
                # Cek apakah data sudah lengkap (ada di kedua tabel)
                if check_date_exists(target_date, check_both_tables=True):
                    logger.info(f"✅ Data lengkap untuk tanggal {today_str} sudah ada di database, skip crawling")
                    print(f"✅ Data lengkap untuk tanggal {today_str} sudah ada di database, skip crawling")
                    return None, None  # Return None untuk menandakan skip
                # Jika hanya salah satu yang ada, tetap crawl untuk melengkapi
                elif check_date_exists(target_date, check_both_tables=False):
                    logger.info(f"⚠️ Data sebagian untuk tanggal {today_str} sudah ada, lanjut crawl untuk melengkapi")
                    print(f"⚠️ Data sebagian untuk tanggal {today_str} sudah ada, lanjut crawl untuk melengkapi")
            except Exception as e:
                logger.warning(f"⚠️ Error checking database: {e}, lanjut crawling...")
        
        logger.info(f"🔄 Memulai crawling data untuk tanggal: {today_str}")
        print(f"📅 Mengambil data untuk tanggal: {today_str}")

        data_dir = Path(__file__).resolve().parents[1] / "data" / "harian"
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Data directory: {data_dir}")
    except Exception as e:
        logger.error(f"❌ Error setup: {e}")
        return None, None

    # --- 1. Crawling Polutan (Open-Meteo)
    logger.info(f"🌬️  Mengambil data polutan dari Open-Meteo API untuk tanggal: {today_str}...")
    df_polut = None
    try:
        air_url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={latitude}&longitude={longitude}&hourly=pm10,pm2_5,ozone,carbon_monoxide"
            f"&timezone=Asia/Bangkok&start_date={today_str}&end_date={today_str}"
        )
        
        response = requests.get(air_url, timeout=30)
        response.raise_for_status()
        
        air_res = response.json()
        if "hourly" in air_res and len(air_res["hourly"]) > 0:
            df_polut = pd.DataFrame(air_res["hourly"])
            
            # Simpan ke CSV
            polutan_file = data_dir / f"polutan_{today_str}.csv"
            df_polut.to_csv(polutan_file, index=False)
            
            logger.info(f"✅ Data polutan tersimpan: {polutan_file}")
            print(f"✅ Data polutan tersimpan: polutan_{today_str}.csv")
        else:
            logger.warning(f"⚠️ Tidak ada data polutan untuk tanggal {today_str}!")
            print(f"⚠️ Tidak ada data polutan untuk tanggal {today_str}!")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error mengambil data polutan: {e}")
        print(f"❌ Error mengambil data polutan: {e}")
    except Exception as e:
        logger.error(f"❌ Error memproses data polutan: {e}")
        print(f"❌ Error memproses data polutan: {e}")

    # --- 2. Crawling Cuaca (Visual Crossing) dengan retry logic
    logger.info(f"🌤️  Mengambil data cuaca dari Visual Crossing API untuk tanggal: {today_str}...")
    df_cuaca = None
    api_key = os.getenv("VISUAL_API_KEY")
    if not api_key:
        error_msg = "⚠️ VISUAL_API_KEY tidak ditemukan di environment variables!"
        logger.error(error_msg)
        print(error_msg)
        print(f"💡 Pastikan file .env ada di: {env_path}")
        return df_polut, None
    
    # Retry logic untuk handle rate limiting
    max_retries = 3
    retry_delay = 5  # Start dengan 5 detik
    
    for attempt in range(max_retries):
        try:
            weather_url = (
                f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
                f"Malang/{today_str}?unitGroup=metric&include=hours&key={api_key}&contentType=csv"
            )
            
            response = requests.get(weather_url, timeout=30)
            
            # Handle rate limiting (429)
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff: 5, 10, 20 detik
                    logger.warning(f"⚠️ Rate limit (429). Menunggu {wait_time} detik sebelum retry...")
                    print(f"⚠️ Rate limit terdeteksi. Menunggu {wait_time} detik...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ Rate limit tetap terjadi setelah {max_retries} attempts")
                    print(f"❌ Rate limit: API terlalu banyak request. Coba lagi nanti.")
                    return df_polut, None
            
            response.raise_for_status()
            
            df_cuaca = pd.read_csv(weather_url)
            
            # Simpan ke CSV
            cuaca_file = data_dir / f"cuaca_{today_str}.csv"
            df_cuaca.to_csv(cuaca_file, index=False)
            
            logger.info(f"✅ Data cuaca tersimpan: {cuaca_file}")
            print(f"✅ Data cuaca tersimpan: cuaca_{today_str}.csv")
            break  # Success, keluar dari loop retry
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"⚠️ Rate limit (429). Menunggu {wait_time} detik sebelum retry...")
                    print(f"⚠️ Rate limit terdeteksi. Menunggu {wait_time} detik...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ Rate limit tetap terjadi setelah {max_retries} attempts")
                    print(f"❌ Rate limit: API terlalu banyak request. Coba lagi nanti.")
                    return df_polut, None
            else:
                logger.error(f"❌ HTTP Error mengambil data cuaca: {e}")
                print(f"❌ Error mengambil data cuaca: {e}")
                break
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error mengambil data cuaca: {e}")
            print(f"❌ Error mengambil data cuaca: {e}")
            break
        except Exception as e:
            logger.error(f"❌ Error memproses data cuaca: {e}")
            print(f"❌ Error memproses data cuaca: {e}")
            break
    
    # --- 3. Simpan ke database jika diminta
    if save_to_db and (df_polut is not None or df_cuaca is not None):
        try:
            from crawler.db_handler import save_to_database
            save_to_database(df_polut, df_cuaca, target_date)
            logger.info("✅ Data tersimpan ke database")
            print("✅ Data tersimpan ke database")
        except Exception as e:
            logger.warning(f"⚠️ Error menyimpan ke database: {e}")
            print(f"⚠️ Error menyimpan ke database: {e}")
    
    logger.info("✅ Crawling selesai!")
    print("✅ Crawling selesai!")
    return df_polut, df_cuaca


# ============================================================================
# MAIN - untuk testing langsung
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("DAILY CRAWLER - Test Run")
    print("="*80)
    
    try:
        result = crawl_daily(save_to_db=True)
        if result[0] is not None or result[1] is not None:
            print("\n✅ Crawling berhasil!")
            sys.exit(0)
        else:
            print("\n❌ Crawling gagal!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Crawling dihentikan oleh user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}", exc_info=True)
        print(f"\n❌ Error fatal: {e}")
        sys.exit(1)
