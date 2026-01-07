# backend/crawler/bulk_crawler.py
"""
Bulk crawler untuk mengambil data 30 hari ke belakang dan menyimpan ke database.
"""
import sys
from pathlib import Path
from datetime import date, timedelta
import logging
import time

# Fix path untuk import
if __name__ == "__main__":
    backend_dir = Path(__file__).resolve().parents[1]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

from crawler.daily_crawler import crawl_daily

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def bulk_crawl_days(days_back: int = 30, delay: float = 3.0):
    """
    Crawl data untuk beberapa hari ke belakang.
    
    Args:
        days_back (int): Jumlah hari ke belakang dari hari ini (default: 30)
        delay (float): Delay antar request dalam detik (default: 3.0 untuk menghindari rate limit)
    
    Returns:
        dict: Statistik crawling
    """
    today = date.today()
    stats = {
        'total': days_back,
        'success': 0,
        'failed': 0,
        'skipped': 0
    }
    
    print("="*80)
    print(f"BULK CRAWLER - Mengambil data {days_back} hari ke belakang")
    print("="*80)
    print(f"📅 Range: {(today - timedelta(days=days_back-1)).strftime('%Y-%m-%d')} hingga {today.strftime('%Y-%m-%d')}")
    print()
    
    for i in range(days_back):
        target_date = today - timedelta(days=i)
        target_str = target_date.strftime("%Y-%m-%d")
        
        # Cek apakah data sudah lengkap di database sebelum crawling
        try:
            from crawler.db_handler import check_date_exists
            
            if check_date_exists(target_date, check_both_tables=True):
                stats['skipped'] += 1
                print(f"[{i+1}/{days_back}] ⏭️  Skip {target_str} - Data lengkap sudah ada di database")
                logger.info(f"[{i+1}/{days_back}] Skip {target_str} - Data lengkap sudah ada di database")
                # Tetap delay sedikit untuk konsistensi
                if i < days_back - 1:
                    time.sleep(0.5)
                continue
        except Exception as e:
            logger.warning(f"⚠️ Error checking database: {e}, lanjut crawling...")
        
        # Data belum lengkap atau belum ada, lanjut crawling
        try:
            print(f"[{i+1}/{days_back}] Crawling data untuk tanggal: {target_str}")
            logger.info(f"[{i+1}/{days_back}] Crawling data untuk tanggal: {target_str}")
            
            df_polut, df_cuaca = crawl_daily(target_date=target_date, save_to_db=True, skip_if_exists=True)
            
            # Jika return None, None berarti skip (karena sudah dicek lengkap di atas, berarti data belum lengkap)
            # Jika return dataframe, berarti berhasil
            if df_polut is not None or df_cuaca is not None:
                stats['success'] += 1
                print(f"✅ Berhasil: {target_str}")
            elif df_polut is None and df_cuaca is None:
                # Bisa berarti skip atau gagal total, tapi karena sudah dicek di atas, kemungkinan gagal
                # Cek lagi apakah sekarang sudah ada data
                from crawler.db_handler import check_date_exists
                if check_date_exists(target_date, check_both_tables=True):
                    stats['skipped'] += 1
                    print(f"⏭️  Skip: {target_str} (data sudah ada)")
                else:
                    stats['failed'] += 1
                    print(f"❌ Gagal: {target_str}")
        
        except Exception as e:
            stats['failed'] += 1
            logger.error(f"❌ Error crawling {target_str}: {e}")
            print(f"❌ Error: {e}")
        
        # Delay antar request untuk menghindari rate limiting
        # Delay lebih lama untuk Visual Crossing API (free tier rate limit)
        if i < days_back - 1:
            print(f"⏳ Menunggu {delay} detik sebelum request berikutnya...")
            time.sleep(delay)
    
    print("\n" + "="*80)
    print("BULK CRAWLING SELESAI")
    print("="*80)
    print(f"Total: {stats['total']} hari")
    print(f"✅ Berhasil: {stats['success']}")
    print(f"❌ Gagal: {stats['failed']}")
    print(f"⏭️  Skipped: {stats['skipped']}")
    print("="*80)
    
    return stats


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Bulk crawler untuk data historis')
    parser.add_argument('--days', type=int, default=30,
                       help='Jumlah hari ke belakang (default: 30)')
    parser.add_argument('--delay', type=float, default=3.0,
                       help='Delay antar request dalam detik (default: 3.0 untuk menghindari rate limit)')
    
    args = parser.parse_args()
    
    try:
        stats = bulk_crawl_days(days_back=args.days, delay=args.delay)
        
        if stats['success'] > 0:
            print(f"\n✅ Bulk crawling selesai! {stats['success']} hari berhasil di-crawl.")
            sys.exit(0)
        else:
            print("\n❌ Tidak ada data yang berhasil di-crawl.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Bulk crawling dihentikan oleh user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}", exc_info=True)
        print(f"\n❌ Error fatal: {e}")
        sys.exit(1)

