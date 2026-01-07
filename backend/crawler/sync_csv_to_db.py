"""
Sinkronisasi data CSV harian ke database SQLite.

File ini membaca semua file CSV dari direktori data/harian dan
menyimpan data yang belum ada ke database.
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import date, datetime
import re
import logging
import argparse

# Fix path for import
if __name__ == "__main__":
    backend_dir = Path(__file__).resolve().parents[1]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

from crawler.db_handler import init_database, check_date_exists, save_to_database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_date_from_filename(filename: str) -> date | None:
    """
    Mengekstrak tanggal dari nama file CSV.
    
    Format yang didukung:
    - polutan_YYYY-MM-DD.csv
    - cuaca_YYYY-MM-DD.csv
    
    Args:
        filename: Nama file (dengan atau tanpa path)
    
    Returns:
        date object atau None jika tidak ditemukan
    """
    # Extract date pattern YYYY-MM-DD
    pattern = r'(\d{4}-\d{2}-\d{2})'
    match = re.search(pattern, filename)
    
    if match:
        try:
            date_str = match.group(1)
            return datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            logger.warning(f"⚠️ Invalid date format in filename: {filename}")
            return None
    return None


def sync_csv_files_to_db(data_dir: Path = None, force: bool = False):
    """
    Sinkronisasi semua file CSV dari direktori ke database.
    
    Args:
        data_dir: Path ke direktori data/harian (default: backend/data/harian)
        force: Jika True, akan overwrite data yang sudah ada (default: False)
    
    Returns:
        dict: Statistik sinkronisasi
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "data" / "harian"
    
    if not data_dir.exists():
        logger.error(f"❌ Direktori tidak ditemukan: {data_dir}")
        return {
            'total_files': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'errors': []
        }
    
    # Inisialisasi database
    init_database()
    
    stats = {
        'total_files': 0,
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'errors': []
    }
    
    # Cari semua file CSV
    csv_files = list(data_dir.glob("*.csv"))
    stats['total_files'] = len(csv_files)
    
    if stats['total_files'] == 0:
        logger.warning(f"⚠️ Tidak ada file CSV ditemukan di {data_dir}")
        return stats
    
    print("="*80)
    print(f"SINKRONISASI CSV KE DATABASE")
    print("="*80)
    print(f"📁 Direktori: {data_dir}")
    print(f"📄 Total file CSV: {stats['total_files']}")
    print(f"🔄 Mode: {'Force (overwrite)' if force else 'Skip existing'}")
    print("="*80)
    print()
    
    # Proses setiap file CSV
    for idx, csv_file in enumerate(sorted(csv_files), 1):
        filename = csv_file.name
        target_date = extract_date_from_filename(filename)
        
        if target_date is None:
            logger.warning(f"⚠️ [{idx}/{stats['total_files']}] Tidak dapat mengekstrak tanggal dari: {filename}")
            stats['failed'] += 1
            stats['errors'].append(f"{filename}: Invalid date format")
            continue
        
        logger.info(f"[{idx}/{stats['total_files']}] Memproses: {filename} (tanggal: {target_date})")
        
        # Cek apakah data sudah ada (jika tidak force mode)
        if not force:
            if check_date_exists(target_date, check_both_tables=False):
                logger.info(f"✅ Data untuk {target_date} sudah ada di database, skipping...")
                stats['skipped'] += 1
                continue
        
        try:
            # Baca file CSV
            df = pd.read_csv(csv_file)
            
            if df.empty:
                logger.warning(f"⚠️ File kosong: {filename}")
                stats['failed'] += 1
                stats['errors'].append(f"{filename}: File is empty")
                continue
            
            # Tentukan tipe data berdasarkan nama file
            if filename.startswith('polutan_'):
                # Data polutan
                df_polut = df.copy()
                df_cuaca = None
                
                # Pastikan kolom time ada
                if 'time' not in df_polut.columns:
                    logger.warning(f"⚠️ Kolom 'time' tidak ditemukan di {filename}")
                    stats['failed'] += 1
                    stats['errors'].append(f"{filename}: Missing 'time' column")
                    continue
                
            elif filename.startswith('cuaca_'):
                # Data cuaca
                df_polut = None
                df_cuaca = df.copy()
                
                # Pastikan kolom datetime ada
                if 'datetime' not in df_cuaca.columns:
                    logger.warning(f"⚠️ Kolom 'datetime' tidak ditemukan di {filename}")
                    stats['failed'] += 1
                    stats['errors'].append(f"{filename}: Missing 'datetime' column")
                    continue
            else:
                logger.warning(f"⚠️ Format file tidak dikenali: {filename} (harus polutan_ atau cuaca_)")
                stats['failed'] += 1
                stats['errors'].append(f"{filename}: Unknown file format")
                continue
            
            # Simpan ke database
            save_to_database(df_polut=df_polut, df_cuaca=df_cuaca, target_date=target_date)
            
            logger.info(f"✅ Berhasil menyimpan data untuk {target_date}")
            stats['processed'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error memproses {filename}: {e}", exc_info=True)
            stats['failed'] += 1
            stats['errors'].append(f"{filename}: {str(e)}")
    
    # Print summary
    print()
    print("="*80)
    print("SINKRONISASI SELESAI")
    print("="*80)
    print(f"Total file:        {stats['total_files']}")
    print(f"Berhasil:          {stats['processed']}")
    print(f"Skipped (existing): {stats['skipped']}")
    print(f"Gagal:             {stats['failed']}")
    print("="*80)
    
    if stats['errors']:
        print("\n⚠️ Error details:")
        for error in stats['errors']:
            print(f"  - {error}")
    
    return stats


def main():
    """Entry point untuk command line"""
    parser = argparse.ArgumentParser(
        description='Sinkronisasi data CSV harian ke database SQLite'
    )
    parser.add_argument(
        '--dir',
        type=str,
        default=None,
        help='Path ke direktori data/harian (default: backend/data/harian)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite data yang sudah ada di database'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Sinkronisasi hanya untuk tanggal tertentu (format: YYYY-MM-DD)'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.dir) if args.dir else None
    
    # Jika --date diberikan, hanya proses file untuk tanggal tersebut
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
            if data_dir is None:
                data_dir = Path(__file__).resolve().parents[1] / "data" / "harian"
            
            # Cari file untuk tanggal tersebut
            polutan_file = data_dir / f"polutan_{args.date}.csv"
            cuaca_file = data_dir / f"cuaca_{args.date}.csv"
            
            stats = {
                'total_files': 0,
                'processed': 0,
                'skipped': 0,
                'failed': 0,
                'errors': []
            }
            
            init_database()
            
            files_to_process = []
            if polutan_file.exists():
                files_to_process.append(polutan_file)
            if cuaca_file.exists():
                files_to_process.append(cuaca_file)
            
            if not files_to_process:
                print(f"❌ Tidak ada file CSV untuk tanggal {args.date}")
                return
            
            stats['total_files'] = len(files_to_process)
            
            for csv_file in files_to_process:
                try:
                    df = pd.read_csv(csv_file)
                    filename = csv_file.name
                    
                    if filename.startswith('polutan_'):
                        df_polut = df
                        df_cuaca = None
                    elif filename.startswith('cuaca_'):
                        df_polut = None
                        df_cuaca = df
                    else:
                        continue
                    
                    if not args.force and check_date_exists(target_date, check_both_tables=False):
                        print(f"✅ Data untuk {target_date} sudah ada, skipping... (gunakan --force untuk overwrite)")
                        stats['skipped'] += 1
                        continue
                    
                    save_to_database(df_polut=df_polut, df_cuaca=df_cuaca, target_date=target_date)
                    stats['processed'] += 1
                    print(f"✅ Berhasil menyimpan data untuk {target_date}")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    stats['failed'] += 1
                    stats['errors'].append(f"{csv_file.name}: {str(e)}")
            
            print(f"\n✅ Selesai: {stats['processed']} berhasil, {stats['skipped']} skipped, {stats['failed']} gagal")
            
        except ValueError:
            print(f"❌ Format tanggal tidak valid. Gunakan format: YYYY-MM-DD")
            return
    else:
        # Proses semua file
        sync_csv_files_to_db(data_dir=data_dir, force=args.force)


if __name__ == "__main__":
    main()

