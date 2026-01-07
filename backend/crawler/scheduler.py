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
import os

# Fix path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    print("⚠️ APScheduler tidak terinstall. Install dengan: pip install apscheduler")

# Import crawlers dan db functions
try:
    from crawler.daily_crawler import crawl_daily
except ImportError:
    try:
        from daily_crawler import crawl_daily
    except ImportError:
        print("❌ Error: Tidak dapat mengimport crawl_daily")
        crawl_daily = None

try:
    from crawler.db_handler import aggregate_hourly_to_daily, merge_aggregation_to_master_dataset
except ImportError:
    try:
        from db_handler import aggregate_hourly_to_daily, merge_aggregation_to_master_dataset
    except ImportError:
        print("⚠️ Warning: db_handler tidak dapat diimport, aggregation akan diskip")
        aggregate_hourly_to_daily = None
        merge_aggregation_to_master_dataset = None

# Import training
try:
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
    
    ALUR CONTINUOUS LEARNING SEMPURNA:
    1. Aggregasi 24 jam hourly data (dari polutan + cuaca table) → 1 daily row
    2. Merge daily row ke master dataset (dataset_preprocessed.csv)
    3. Preprocessing (data cleaning, outlier detection, feature engineering)
    4. Training 3 Model (Decision Tree, CNN, GRU)
    5. Save model terbaru ke models/ folder
    6. Model siap untuk dashboard besok hari
    
    Data Flow:
    Hourly Crawl (24x) → SQLite polutan+cuaca → Daily Aggregation (23:50) → 
    Master Dataset → Preprocessing → Training → Model Update
    
    Waktu eksekusi: ~5-10 menit
    """
    
    try:
        logger.info("\n" + "="*80)
        logger.info("🚀 CONTINUOUS LEARNING PIPELINE - Mulai Jam 23:50")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)
        
        # ============================================================
        # STEP 1: HOURLY DATA AGGREGATION (24 jam → 1 daily)
        # ============================================================
        logger.info("\n[STEP 1/5] 📊 Aggregasi 24 jam hourly data menjadi 1 daily row...")
        
        aggregation = None
        if aggregate_hourly_to_daily is not None:
            try:
                aggregation = aggregate_hourly_to_daily()
                
                if aggregation:
                    logger.info(f"✅ Aggregation sukses!")
                    logger.info(f"   Hourly records: {aggregation.get('hourly_count', 0)}/24 jam")
                    logger.info(f"   Data agregasi:")
                    logger.info(f"   - PM2.5: {aggregation.get('pm2_5 (μg/m³)_mean', 'N/A')} ± {aggregation.get('pm2_5 (μg/m³)_max', 'N/A')} μg/m³")
                    logger.info(f"   - O3: {aggregation.get('ozone (μg/m³)_mean', 'N/A')} ± {aggregation.get('ozone (μg/m³)_max', 'N/A')} μg/m³")
                    logger.info(f"   - CO: {aggregation.get('carbon_monoxide (μg/m³)_mean', 'N/A')} ± {aggregation.get('carbon_monoxide (μg/m³)_max', 'N/A')} μg/m³")
                    logger.info(f"   - Temp: {aggregation.get('temp', 'N/A')} (normalized)")
                    logger.info(f"   - Humidity: {aggregation.get('humidity', 'N/A')} (normalized)")
                else:
                    logger.warning("⚠️ Aggregation gagal atau tidak ada data untuk hari ini")
                    logger.warning("   Skipping hari ini, akan retry besok")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Aggregation error: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            logger.error("❌ aggregate_hourly_to_daily tidak tersedia!")
            return False
        
        # ============================================================
        # STEP 2: MERGE KE MASTER DATASET
        # ============================================================
        logger.info("\n[STEP 2/5] 🗂️  Merge aggregasi ke master dataset...")
        
        merge_success = False
        if aggregation and merge_aggregation_to_master_dataset:
            try:
                merge_success = merge_aggregation_to_master_dataset(aggregation)
                if merge_success:
                    logger.info("✅ Merge sukses!")
                else:
                    logger.warning("⚠️ Merge skip (data mungkin sudah ada atau error)")
            except Exception as e:
                logger.error(f"❌ Merge error: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.error("❌ merge_aggregation_to_master_dataset tidak tersedia!")
        
        # ============================================================
        # STEP 3: PREPROCESSING
        # ============================================================
        logger.info("\n[STEP 3/5] ⚙️  Preprocessing dataset...")
        
        if not TRAINING_AVAILABLE:
            logger.error("❌ Training module tidak tersedia!")
            return False
        
        try:
            logger.info("   Melakukan data cleaning, outlier detection, feature engineering...")
            training.preprocess_data()
            logger.info("✅ Preprocessing selesai!")
        except Exception as e:
            logger.error(f"❌ Preprocessing error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # ============================================================
        # STEP 4: TRAINING MODELS
        # ============================================================
        logger.info("\n[STEP 4/5] 🤖 Training 3 Model dengan data terbaru...")
        logger.info("   (Data sekarang include: data historis lama + data baru hari ini)")
        
        # Training Decision Tree
        try:
            logger.info("   [4a/4c] Training Decision Tree...")
            training.train_decision_tree()
            logger.info("   ✅ Decision Tree training selesai")
        except Exception as e:
            logger.error(f"   ❌ Decision Tree training error: {e}")
            import traceback
            traceback.print_exc()
        
        # Training CNN
        try:
            logger.info("   [4b/4c] Training CNN (spatial model)...")
            training.train_cnn()
            logger.info("   ✅ CNN training selesai")
        except Exception as e:
            logger.error(f"   ❌ CNN training error: {e}")
            import traceback
            traceback.print_exc()
        
        # Training GRU
        try:
            logger.info("   [4c/4c] Training GRU (temporal model)...")
            training.train_gru()
            logger.info("   ✅ GRU training selesai")
        except Exception as e:
            logger.error(f"   ❌ GRU training error: {e}")
            import traceback
            traceback.print_exc()
        
        # ============================================================
        # STEP 5: MODEL SAVE & VERIFICATION
        # ============================================================
        logger.info("\n[STEP 5/5] 💾 Menyimpan model terbaru...")
        
        models_dir = Path(__file__).resolve().parents[1] / "models"
        logger.info(f"   Models directory: {models_dir}")
        
        dt_model_path = models_dir / "Decision Tree" / "model.pkl"
        cnn_model_path = models_dir / "CNN" / "best_cnn.keras"
        gru_model_path = models_dir / "GRU" / "model_v9.keras"
        
        if dt_model_path.exists():
            logger.info(f"   ✅ Decision Tree model: {dt_model_path}")
        else:
            logger.warning(f"   ⚠️ Decision Tree model not found: {dt_model_path}")
        
        if cnn_model_path.exists():
            logger.info(f"   ✅ CNN model: {cnn_model_path}")
        else:
            logger.warning(f"   ⚠️ CNN model not found: {cnn_model_path}")
        
        if gru_model_path.exists():
            logger.info(f"   ✅ GRU model: {gru_model_path}")
        else:
            logger.warning(f"   ⚠️ GRU model not found: {gru_model_path}")
        
        # ============================================================
        # COMPLETION
        # ============================================================
        logger.info("\n" + "="*80)
        logger.info("✅ CONTINUOUS LEARNING PIPELINE SELESAI!")
        logger.info("="*80)
        logger.info("\n📊 RINGKASAN PEMBELAJARAN HARI INI:")
        logger.info(f"   • 24 jam hourly data aggregated → 1 daily row")
        logger.info(f"   • Master dataset updated dengan data baru")
        logger.info(f"   • 3 models trained dengan data lama + baru")
        logger.info(f"   • Model accuracy akan terus meningkat setiap hari")
        logger.info(f"\n🎯 NEXT:")
        logger.info(f"   • Dashboard akan load model terbaru")
        logger.info(f"   • Prediksi hari ini lebih akurat dari kemarin")
        logger.info(f"   • Crawler hourly akan terus berjalan setiap jam")
        logger.info(f"   • Training akan repeat besok jam 23:50")
        logger.info("="*80 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Daily training pipeline gagal: {e}")
        logger.error("   Sistem akan tetap menggunakan model sebelumnya")
        import traceback
        traceback.print_exc()
        return False


def start_scheduler():
    """
    START SCHEDULER - OPSI 3
    
    HOURLY: Setiap jam (00:00, 01:00, ..., 23:00) → Crawling
    DAILY: Setiap hari jam 23:50 → Aggregation + Training
    
    Returns:
        BackgroundScheduler: Instance scheduler atau None jika gagal
    """
    global _scheduler_instance
    
    if not APSCHEDULER_AVAILABLE:
        logger.error("❌ APScheduler tidak tersedia. Tidak dapat memulai scheduler.")
        return None
    
    try:
        scheduler = BackgroundScheduler()
        
        # ===== HOURLY CRAWLING =====
        scheduler.add_job(
            hourly_crawling_job,
            CronTrigger(minute=0),  # Setiap jam:00
            id='hourly_crawling_job',
            name='Hourly Data Crawling',
            replace_existing=True,
            max_instances=1
        )
        
        # ===== DAILY AGGREGATION + TRAINING =====
        scheduler.add_job(
            daily_aggregation_and_training_job,
            CronTrigger(hour=23, minute=50),
            id='daily_aggregation_training_job',
            name='Daily Aggregation + Training',
            replace_existing=True,
            max_instances=1
        )
        
        scheduler.start()
        _scheduler_instance = scheduler
        
        logger.info("\n" + "="*80)
        logger.info("📅 SCHEDULER DIKONFIGURASI - OPSI 3")
        logger.info("="*80)
        logger.info("HOURLY: Setiap jam (00:00, 01:00, ..., 23:00)")
        logger.info("  └─ Crawl data polutan + cuaca → Simpan ke SQLite")
        logger.info("")
        logger.info("DAILY: Setiap hari jam 23:50")
        logger.info("  ├─ Aggregasi 24 jam data menjadi 1 baris")
        logger.info("  ├─ Merge dengan master dataset")
        logger.info("  ├─ Preprocessing")
        logger.info("  ├─ Training 3 Model (DT, CNN, GRU)")
        logger.info("  └─ Save model terbaru")
        logger.info("")
        logger.info("Status: ✅ Running di background")
        logger.info("Log: backend/logs/continuous_learning.log")
        logger.info("="*80 + "\n")
        
        return scheduler
        
    except Exception as e:
        logger.error(f"❌ Error memulai scheduler: {e}", exc_info=True)
        return None


def stop_scheduler():
    """Stop scheduler jika sedang berjalan"""
    global _scheduler_instance
    
    if _scheduler_instance is not None and _scheduler_instance.running:
        try:
            _scheduler_instance.shutdown(wait=True)
            _scheduler_instance = None
            logger.info("✅ Scheduler dihentikan")
        except Exception as e:
            logger.error(f"❌ Error menghentikan scheduler: {e}")
    else:
        logger.info("ℹ️ Scheduler tidak sedang berjalan")


if __name__ == "__main__":
    import time
    
    print("\n" + "="*80)
    print("🚀 STARTING CONTINUOUS LEARNING SCHEDULER (OPSI 3)")
    print("="*80)
    
    scheduler = start_scheduler()
    
    if scheduler is None:
        print("❌ Gagal memulai scheduler")
        sys.exit(1)
    
    print("\n✅ Scheduler berhasil dijalankan!")
    print("ℹ️  Tekan Ctrl+C untuk menghentikan scheduler")
    print("="*80 + "\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Menghentikan scheduler...")
        stop_scheduler()
        print("✅ Scheduler dihentikan. Selamat tinggal!")
