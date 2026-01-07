#!/usr/bin/env python3
"""
Script untuk manual crawling data yang terlewat
Gunakan ini untuk backfill data tanggal 3 dan 4 Januari 2026
"""
import sys
from pathlib import Path
from datetime import date, datetime, timedelta
import logging

# Setup path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from crawler.daily_crawler import crawl_daily
from crawler.db_handler import aggregate_hourly_to_daily, merge_aggregation_to_master_dataset

def backfill_missing_dates(start_date: date, end_date: date):
    """
    Crawl dan process data untuk range tanggal tertentu
    
    Args:
        start_date: Tanggal awal (format: date object atau "2026-01-03")
        end_date: Tanggal akhir (format: date object atau "2026-01-04")
    """
    
    # Parse dates jika string
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    logger.info("=" * 80)
    logger.info(f"🔄 BACKFILL MISSING DATA: {start_date} hingga {end_date}")
    logger.info("=" * 80)
    
    current_date = start_date
    success_count = 0
    fail_count = 0
    
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📅 Processing tanggal: {date_str}")
        logger.info(f"{'='*80}")
        
        try:
            # Step 1: Crawl
            logger.info(f"[1/3] 🌐 Crawling data untuk {date_str}...")
            result = crawl_daily(target_date=current_date, save_to_db=True, skip_if_exists=False)
            
            if result == (None, None):
                logger.warning(f"⚠️ Crawling skip (data sudah lengkap atau gagal) untuk {date_str}")
                fail_count += 1
            else:
                logger.info(f"✅ Crawling sukses untuk {date_str}")
                
                # Step 2: Aggregate
                logger.info(f"[2/3] 📊 Aggregasi 24 jam menjadi 1 daily row...")
                aggregation = aggregate_hourly_to_daily()
                
                if aggregation:
                    logger.info(f"✅ Aggregasi sukses!")
                    logger.info(f"   PM2.5: {aggregation.get('pm2_5 (μg/m³)_mean', 'N/A')} μg/m³")
                    logger.info(f"   O3: {aggregation.get('ozone (μg/m³)_mean', 'N/A')} μg/m³")
                    logger.info(f"   CO: {aggregation.get('carbon_monoxide (μg/m³)_mean', 'N/A')} μg/m³")
                    
                    # Step 3: Merge ke dataset
                    logger.info(f"[3/3] 🗂️  Merge ke master dataset...")
                    merge_success = merge_aggregation_to_master_dataset(aggregation)
                    
                    if merge_success:
                        logger.info(f"✅ Merge sukses!")
                        success_count += 1
                    else:
                        logger.warning(f"⚠️ Merge skip atau gagal")
                        fail_count += 1
                else:
                    logger.error(f"❌ Aggregasi gagal untuk {date_str}")
                    fail_count += 1
                    
        except Exception as e:
            logger.error(f"❌ Error processing {date_str}: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
        
        current_date += timedelta(days=1)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ BACKFILL SELESAI")
    logger.info(f"{'='*80}")
    logger.info(f"Sukses: {success_count}, Gagal: {fail_count}")
    logger.info(f"Total hari: {(end_date - start_date).days + 1}")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backfill missing data untuk continuous learning")
    parser.add_argument(
        "--start",
        type=str,
        default="2026-01-03",
        help="Tanggal awal (format: YYYY-MM-DD, default: 2026-01-03)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2026-01-04",
        help="Tanggal akhir (format: YYYY-MM-DD, default: 2026-01-04)"
    )
    
    args = parser.parse_args()
    
    try:
        backfill_missing_dates(args.start, args.end)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)
