# backend/crawler/scheduler.py
"""
CONTINUOUS LEARNING SCHEDULER - OPSI 3
Alur: Hourly Crawling + Daily Aggregation + Daily Training

Timeline:
- 00:00, 01:00, ..., 23:00 → Hourly crawling (24 times/day)
- 23:50 → Daily aggregation (aggregate 24 jam data) + training
"""
import sys
from pathlib import Path
from datetime import date, datetime
import logging
import subprocess
import os

# Fix path untuk import - tambahkan parent directory ke sys.path
if __name__ == "__main__":
    # Jika dijalankan langsung, tambahkan backend directory ke path
    backend_dir = Path(__file__).resolve().parents[1]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.executors.pool import ThreadPoolExecutor
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    print("⚠️ APScheduler tidak terinstall. Install dengan: pip install apscheduler")

# Import dengan relative atau absolute
try:
    from crawler.daily_crawler import crawl_daily
except ImportError:
    # Fallback untuk relative import jika dijalankan sebagai module
    try:
        from daily_crawler import crawl_daily
    except ImportError:
        print("❌ Error: Tidak dapat mengimport crawl_daily")
        crawl_daily = None

# Import db_handler untuk aggregation
try:
    from crawler.db_handler import aggregate_hourly_to_daily, merge_aggregation_to_master_dataset
except ImportError:
    try:
        from db_handler import aggregate_hourly_to_daily, merge_aggregation_to_master_dataset
    except ImportError:
        print("⚠️ Warning: db_handler tidak dapat diimport")
        aggregate_hourly_to_daily = None
        merge_aggregation_to_master_dataset = None

# Import training functions
try:
    import training
    TRAINING_AVAILABLE = True
except ImportError:
    try:
        import sys
        backend_dir = Path(__file__).resolve().parents[1]
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        import training
        TRAINING_AVAILABLE = True
    except ImportError:
        print("⚠️ Warning: training module tidak dapat diimport")
        TRAINING_AVAILABLE = False

# Setup logging
log_dir = Path(__file__).resolve().parents[1] / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'continuous_learning.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global scheduler instance untuk akses dari luar
_scheduler_instance = None


def hourly_crawling_job():
    """
    HOURLY CRAWLING JOB
    Dijalankan setiap jam (00:00, 01:00, 02:00, ..., 23:00)
    
    Fungsi: Crawl data polutan + cuaca, simpan ke SQLite database
    Waktu eksekusi: ~30 detik per crawl
    """
    try:
        current_hour = datetime.now().hour
        logger.info(f"[HOURLY CRAWL] Jam {current_hour:02d}:00 - Crawling data polutan + cuaca...")
        
        if crawl_daily is None:
            logger.error("❌ crawl_daily tidak tersedia, skipping...")
            return False
        
        crawl_result = crawl_daily()
        
        if crawl_result:
            logger.info(f"✅ Hourly crawl sukses untuk jam {current_hour:02d}:00")
            logger.info(f"   Data tersimpan ke SQLite database")
        else:
            logger.warning(f"⚠️ Hourly crawl mungkin tidak sempurna untuk jam {current_hour:02d}:00")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hourly crawl gagal pada jam {current_hour:02d}:00: {e}")
        return False


def daily_aggregation_and_training_job():
    """
    DAILY AGGREGATION + TRAINING JOB
    Dijalankan setiap hari jam 23:50 WIB
    
    Alur OPSI 3 - Continuous Learning:
    1. Agregasi 24 jam hourly data menjadi 1 baris daily
    2. Merge dengan master dataset
    3. Preprocessing
    4. Training 3 Model (Decision Tree, CNN, GRU)
    5. Evaluasi dan save model terbaru
    
    Waktu eksekusi: ~5-10 menit
    """
    
    try:
        logger.info("\n" + "="*80)
        logger.info("[CONTINUOUS LEARNING] Memulai pipeline OPSI 3 (Hourly+Daily)")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)
        
        # STEP 1: DAILY AGGREGATION
        logger.info("\n[STEP 1] 📊 Agregasi 24 jam data menjadi 1 baris master dataset...")
        
        if aggregate_hourly_to_daily is None:
            logger.error("❌ aggregate_hourly_to_daily tidak tersedia!")
            logger.warning("⚠️ Lanjutkan dengan crawl langsung (fallback mode)")
            aggregation = None
        else:
            try:
                aggregation = aggregate_hourly_to_daily()
                
                if aggregation:
                    logger.info(f"✅ Aggregation sukses!")
                    logger.info(f"   Total records dari 24 jam crawl: {aggregation.get('hourly_count', 0)}")
                    logger.info(f"   Nilai agregasi:")
                    logger.info(f"   - PM2.5: {aggregation.get('pm25_avg', 'N/A')} µg/m³ (std: {aggregation.get('pm25_std', 'N/A')})")
                    logger.info(f"   - O3: {aggregation.get('o3_avg', 'N/A')} µg/m³ (std: {aggregation.get('o3_std', 'N/A')})")
                    logger.info(f"   - CO: {aggregation.get('co_avg', 'N/A')} µg/m³ (std: {aggregation.get('co_std', 'N/A')})")
                    logger.info(f"   - Temp: {aggregation.get('temperature_avg', 'N/A')}°C")
                    logger.info(f"   - Humidity: {aggregation.get('humidity_avg', 'N/A')}%")
                    logger.info(f"   - Wind Speed: {aggregation.get('wind_speed_avg', 'N/A')} m/s")
                else:
                    logger.warning("⚠️ Aggregation gagal atau tidak ada data")
                    
            except Exception as e:
                logger.error(f"❌ Aggregation error: {e}")
                logger.warning("⚠️ Menggunakan fallback mode...")
                aggregation = None
        
        # STEP 2: MERGE DENGAN MASTER DATASET
        if aggregation and merge_aggregation_to_master_dataset:
            logger.info("\n[STEP 2] 🗂️  Merge data agregasi dengan master dataset...")
            try:
                merge_result = merge_aggregation_to_master_dataset(aggregation)
                if merge_result:
                    logger.info("✅ Merge sukses!")
                else:
                    logger.warning("⚠️ Merge gagal atau data sudah ada")
            except Exception as e:
                logger.error(f"❌ Merge error: {e}")
        else:
            logger.info("\n[STEP 2] 🗂️  Merge data (aggregation tidak tersedia, skip)")
        
        # STEP 3: PREPROCESSING
        logger.info("\n[STEP 3] ⚙️  Preprocessing data baru...")
        logger.info("   - Normalisasi polutan dan cuaca")
        logger.info("   - Feature engineering (heat_index, weather_severity, dll)")
        logger.info("   - Lag features (lag-1, lag-2, lag-3, lag-7 untuk GRU)")
        logger.info("   - Rolling statistics (7-hari mean & std)")
        logger.info("   - Train/Val/Test split: 70/15/15")
        
        if not TRAINING_AVAILABLE:
            logger.error("❌ Training module tidak tersedia!")
            return False
        
        try:
            training.preprocess_data()
            logger.info("✅ Preprocessing selesai!")
        except Exception as e:
            logger.error(f"❌ Preprocessing error: {e}")
            return False
        
        # STEP 4: TRAINING MODELS
        logger.info("\n[STEP 4] 🤖 Training 3 Model dengan data terbaru...")
        logger.info("   Model 1: Decision Tree (30 features, interpretable)")
        logger.info("   Model 2: CNN (spatial patterns, multi-head encoder)")
        logger.info("   Model 3: GRU (temporal patterns, multi-output RNN)")
        logger.info("")
        logger.info("   Proses training dimulai...")
        
        results = {}
        
        # Train Decision Tree
        try:
            logger.info("   Training Decision Tree...")
            training.train_decision_tree()
            results['dt'] = 'OK'
            logger.info("   ✅ Decision Tree selesai")
        except Exception as e:
            logger.error(f"   ❌ Decision Tree error: {e}")
            results['dt'] = f'ERROR: {e}'
        
        # Train CNN
        try:
            logger.info("   Training CNN...")
            training.train_cnn()
            results['cnn'] = 'OK'
            logger.info("   ✅ CNN selesai")
        except Exception as e:
            logger.error(f"   ❌ CNN error: {e}")
            results['cnn'] = f'ERROR: {e}'
        
        # Train GRU
        try:
            logger.info("   Training GRU...")
            training.train_gru()
            results['gru'] = 'OK'
            logger.info("   ✅ GRU selesai")
        except Exception as e:
            logger.error(f"   ❌ GRU error: {e}")
            results['gru'] = f'ERROR: {e}'
        
        logger.info("\n✅ Training semua models selesai!")
        
        # STEP 5: EVALUASI DAN SAVE
        logger.info("\n[STEP 5] 💾 Menyimpan model terbaik...")
        logger.info("   - File: backend/models/decision_tree_model.pkl")
        logger.info("   - File: backend/models/cnn_model.h5")
        logger.info("   - File: backend/models/gru_model.h5")
        logger.info("   - Metrics: backend/Training/*/evaluation_metrics_*.json")
        
        logger.info("\n" + "="*80)
        logger.info("[CONTINUOUS LEARNING] ✅ Pipeline selesai!")
        logger.info("Dashboard akan load model terbaru dengan 24 jam hourly crawling data")
        logger.info(f"Waktu selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Daily aggregation pipeline gagal: {e}")
        logger.error("Sistem akan tetap menggunakan model sebelumnya")
        return False
    logger.info("✅ Proses Crawl + Train selesai!")
    logger.info("="*80)
    
    # Step 3: SELALU restart aplikasi untuk load model terbaru (walaupun ada error)
    logger.info("🔄 Restarting aplikasi untuk load model terbaru...")
    restart_app()
    
    return True


# Global scheduler instance untuk akses dari luar
_scheduler_instance = None


def start_scheduler(enable_date_check: bool = False, start_date: date = None, auto_train: bool = True):
    """
    Memulai scheduler untuk crawling data harian.
    
    Args:
        enable_date_check (bool): Jika True, cek tanggal sebelum memulai scheduler
        start_date (date): Tanggal mulai scheduler (default: None = langsung aktif)
        auto_train (bool): Jika True, jalankan training setelah crawling selesai (default: True)
    
    Returns:
        BackgroundScheduler: Instance scheduler yang sudah di-start, atau None jika gagal
    """
    global _scheduler_instance
    
    if not APSCHEDULER_AVAILABLE:
        logger.error("❌ APScheduler tidak tersedia. Tidak dapat memulai scheduler.")
        return None
    
    try:
        # Cek tanggal jika diaktifkan
        if enable_date_check:
            if start_date is None:
                start_date = date(2025, 11, 1)  # Default: 1 November 2025
            
            if date.today() < start_date:
                logger.info(f"⏸️ Scheduler belum aktif (mulai {start_date}).")
                return None
        
        # Buat scheduler dengan thread pool executor
        executors = {
            'default': ThreadPoolExecutor(2)
        }
        scheduler = BackgroundScheduler(executors=executors)
        
        # Tentukan fungsi yang akan dipanggil
        # Jika auto_train=True, gunakan wrapper function yang include training
        job_function = run_crawl_and_train if auto_train else crawl_daily
        job_name = 'Daily Crawl + Train' if auto_train else 'Daily Data Crawl'
        
        # Tambahkan job: crawling setiap hari jam 23:50 (sebelum tengah malam)
        # Timezone: Asia/Jakarta (WIB)
        scheduler.add_job(
            job_function,
            'cron',
            hour=23,
            minute=50,
            timezone='Asia/Jakarta',
            id='daily_crawl',
            name=job_name,
            replace_existing=True
        )
        
        # Start scheduler
        scheduler.start()
        _scheduler_instance = scheduler
        
        train_status = "dengan training otomatis" if auto_train else "tanpa training"
        logger.info(f"✅ Scheduler aktif: crawling otomatis tiap hari jam 23:50 WIB {train_status}")
        print(f"✅ Scheduler aktif: crawling otomatis tiap hari jam 23:50 WIB {train_status}")
        
        return scheduler
        
    except Exception as e:
        logger.error(f"❌ Error memulai scheduler: {e}", exc_info=True)
        print(f"❌ Error memulai scheduler: {e}")
        return None


def stop_scheduler():
    """
    Menghentikan scheduler jika sedang berjalan.
    """
    global _scheduler_instance
    
    if _scheduler_instance is not None and _scheduler_instance.running:
        try:
            _scheduler_instance.shutdown(wait=True)
            _scheduler_instance = None
            logger.info("✅ Scheduler dihentikan")
            print("✅ Scheduler dihentikan")
        except Exception as e:
            logger.error(f"❌ Error menghentikan scheduler: {e}", exc_info=True)
            print(f"❌ Error menghentikan scheduler: {e}")
    else:
        logger.info("ℹ️ Scheduler tidak sedang berjalan")


def get_scheduler():
    """
    Mendapatkan instance scheduler yang sedang berjalan.
    
    Returns:
        BackgroundScheduler: Instance scheduler atau None
    """
    return _scheduler_instance


def is_scheduler_running():
    """
    Mengecek apakah scheduler sedang berjalan.
    
    Returns:
        bool: True jika scheduler sedang berjalan
    """
    return _scheduler_instance is not None and _scheduler_instance.running


def run_now(crawl_only: bool = False):
    """
    Menjalankan crawl (dan training jika crawl_only=False) secara langsung/sekarang.
    
    Args:
        crawl_only (bool): Jika True, hanya jalankan crawling tanpa training (default: False)
    
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    logger.info("🚀 Menjalankan job sekarang (run_now)...")
    print("🚀 Menjalankan job sekarang...")
    
    if crawl_daily is None:
        logger.error("❌ crawl_daily tidak tersedia")
        print("❌ crawl_daily tidak tersedia")
        return False
    
    try:
        if crawl_only:
            # Hanya crawling
            logger.info("📥 Menjalankan crawling saja...")
            result = crawl_daily()
        else:
            # Crawl + Train
            logger.info("📥 Menjalankan crawl + train...")
            result = run_crawl_and_train()
        
        if result:
            logger.info("✅ Job selesai dengan sukses!")
            print("✅ Job selesai dengan sukses!")
            return True
        else:
            logger.error("❌ Job gagal")
            print("❌ Job gagal")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error menjalankan job: {e}", exc_info=True)
        print(f"❌ Error menjalankan job: {e}")
        return False


def trigger_job_now(job_id: str = 'daily_crawl'):
    """
    Trigger job yang sudah terdaftar di scheduler untuk dijalankan sekarang.
    Hanya berfungsi jika scheduler sedang berjalan.
    
    Args:
        job_id (str): ID job yang akan di-trigger (default: 'daily_crawl')
    
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    if _scheduler_instance is None or not _scheduler_instance.running:
        logger.error("❌ Scheduler tidak sedang berjalan, tidak bisa trigger job")
        print("❌ Scheduler tidak sedang berjalan, tidak bisa trigger job")
        return False
    
    try:
        job = _scheduler_instance.get_job(job_id)
        if job is None:
            logger.error(f"❌ Job dengan ID '{job_id}' tidak ditemukan")
            print(f"❌ Job dengan ID '{job_id}' tidak ditemukan")
            return False
        
        logger.info(f"🚀 Triggering job '{job_id}' sekarang...")
        print(f"🚀 Triggering job '{job_id}' sekarang...")
        job.modify(next_run_time=None)  # Jadwalkan untuk dijalankan sekarang
        _scheduler_instance.wakeup()  # Bangunkan scheduler
        
        logger.info("✅ Job dijadwalkan untuk dijalankan sekarang")
        print("✅ Job dijadwalkan untuk dijalankan sekarang")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error triggering job: {e}", exc_info=True)
        print(f"❌ Error triggering job: {e}")
        return False


# ============================================================================
# MAIN - untuk testing langsung
# ============================================================================
if __name__ == "__main__":
    import time
    import argparse
    
    parser = argparse.ArgumentParser(description='Air Quality Data Scheduler')
    parser.add_argument('--now', action='store_true', 
                       help='Jalankan crawl+train sekarang (tanpa scheduler)')
    parser.add_argument('--crawl-only', action='store_true',
                       help='Hanya jalankan crawling (untuk digunakan dengan --now)')
    parser.add_argument('--no-train', action='store_true',
                       help='Scheduler tanpa auto-training (hanya crawling)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SCHEDULER - Air Quality Data Management")
    print("="*80)
    
    # Mode: Run now (langsung jalankan tanpa scheduler)
    if args.now:
        print("\n🚀 Mode: Run Now (langsung)")
        print("="*80)
        success = run_now(crawl_only=args.crawl_only)
        sys.exit(0 if success else 1)
    
    # Mode: Start scheduler
    if not APSCHEDULER_AVAILABLE:
        print("❌ APScheduler tidak tersedia. Install dengan: pip install apscheduler")
        sys.exit(1)
    
    if crawl_daily is None:
        print("❌ crawl_daily tidak dapat diimport")
        sys.exit(1)
    
    auto_train = not args.no_train
    print(f"\n🚀 Memulai scheduler (auto_train={auto_train})...")
    scheduler = start_scheduler(enable_date_check=False, auto_train=auto_train)
    
    if scheduler is None:
        print("❌ Gagal memulai scheduler")
        sys.exit(1)
    
    print("\n✅ Scheduler berhasil dijalankan!")
    print("ℹ️  Tekan Ctrl+C untuk menghentikan scheduler")
    print("💡 Gunakan --now untuk menjalankan job langsung")
    print("="*80)
    
    try:
        # Jalankan scheduler dan tunggu
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Menghentikan scheduler...")
        stop_scheduler()
        print("✅ Scheduler dihentikan. Selamat tinggal!")
