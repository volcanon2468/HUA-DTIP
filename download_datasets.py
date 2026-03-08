import os
import sys
import zipfile
import time

try:
    import requests
except ImportError:
    os.system(f"{sys.executable} -m pip install requests")
    import requests

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "raw")
os.makedirs(RAW, exist_ok=True)


def download_file(url, dest_path, chunk_size=32768):
    print(f"  Downloading: {url}")
    print(f"  Saving to:   {dest_path}")
    resp = requests.get(url, stream=True, timeout=600, allow_redirects=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    start = time.time()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                speed = downloaded / (time.time() - start + 1e-6) / 1e6
                print(f"\r  {pct:5.1f}%  {downloaded/1e6:.1f}/{total/1e6:.1f} MB  ({speed:.1f} MB/s)", end="", flush=True)
    elapsed = time.time() - start
    print(f"\n  Done in {elapsed:.0f}s ({downloaded/1e6:.1f} MB)")
    return dest_path


def extract_zip(zip_path, dest_dir):
    print(f"  Extracting: {zip_path}")
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    n = sum(1 for _, _, files in os.walk(dest_dir) for _ in files)
    print(f"  Extracted {n} files into {dest_dir}")


def is_populated(path):
    if not os.path.exists(path):
        return False
    return sum(1 for _, _, files in os.walk(path) for _ in files) > 0


def download_mhealth():
    print(f"\n{'='*60}")
    print("  [1/6] MHEALTH")
    print(f"{'='*60}")
    dest = os.path.join(RAW, "mhealth")
    if is_populated(dest):
        print("  Already downloaded — skipping")
        return True
    url = "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"
    zip_path = os.path.join(RAW, "mhealth.zip")
    try:
        download_file(url, zip_path)
        extract_zip(zip_path, dest)
        os.remove(zip_path)
        print("  ✓ MHEALTH ready")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_pamap2():
    print(f"\n{'='*60}")
    print("  [2/6] PAMAP2")
    print(f"{'='*60}")
    dest = os.path.join(RAW, "pamap2")
    if is_populated(dest):
        print("  Already downloaded — skipping")
        return True
    url = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
    zip_path = os.path.join(RAW, "pamap2.zip")
    try:
        download_file(url, zip_path)
        extract_zip(zip_path, dest)
        inner_zip = os.path.join(dest, "PAMAP2_Dataset.zip")
        if os.path.exists(inner_zip):
            print("  Extracting inner PAMAP2_Dataset.zip...")
            extract_zip(inner_zip, dest)
        os.remove(zip_path)
        print("  ✓ PAMAP2 ready")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_ppg_4week():
    print(f"\n{'='*60}")
    print("  [3/6] 4-Week PPG/HRV (Figshare)")
    print(f"{'='*60}")
    dest = os.path.join(RAW, "ppg_4week")
    os.makedirs(dest, exist_ok=True)
    if is_populated(dest):
        print("  Already downloaded — skipping")
        return True
    files = [
        ("https://ndownloader.figshare.com/files/52669391", "sensor_hrv.csv"),
        ("https://ndownloader.figshare.com/files/52669385", "sensor_hrv_filtered.csv"),
        ("https://ndownloader.figshare.com/files/52669397", "sleep_diary.csv"),
        ("https://ndownloader.figshare.com/files/52669394", "survey.csv"),
    ]
    success = True
    for url, fname in files:
        fpath = os.path.join(dest, fname)
        if os.path.exists(fpath):
            print(f"  {fname} already exists — skipping")
            continue
        try:
            download_file(url, fpath)
        except Exception as e:
            print(f"  ✗ Failed {fname}: {e}")
            success = False
    if success:
        print("  ✓ PPG/HRV ready (CSV files — no raw_data.zip to save bandwidth)")
    return success


def download_stroke_rehab():
    print(f"\n{'='*60}")
    print("  [4/6] Stroke Rehabilitation IMU (Zenodo)")
    print(f"{'='*60}")
    dest = os.path.join(RAW, "stroke_rehab")
    os.makedirs(dest, exist_ok=True)
    if is_populated(dest):
        print("  Already downloaded — skipping")
        return True
    files = [
        ("https://zenodo.org/api/records/10534055/files/raw.zip/content", "stroke_raw.zip"),
        ("https://zenodo.org/api/records/10534055/files/processed.zip/content", "stroke_processed.zip"),
    ]
    success = True
    for url, fname in files:
        zip_path = os.path.join(RAW, fname)
        try:
            download_file(url, zip_path)
            extract_zip(zip_path, dest)
            os.remove(zip_path)
        except Exception as e:
            print(f"  ✗ Failed {fname}: {e}")
            success = False
    if success:
        print("  ✓ Stroke Rehab ready")
    return success


def download_capture24():
    print(f"\n{'='*60}")
    print("  [5/6] CAPTURE-24 Sample (Zenodo)")
    print(f"{'='*60}")
    dest = os.path.join(RAW, "capture24")
    os.makedirs(dest, exist_ok=True)
    if is_populated(dest):
        print("  Already downloaded — skipping")
        return True
    url = "https://zenodo.org/api/records/7705976/files/capture24_sample.zip/content"
    zip_path = os.path.join(RAW, "capture24_sample.zip")
    try:
        download_file(url, zip_path)
        extract_zip(zip_path, dest)
        os.remove(zip_path)
        print("  ✓ CAPTURE-24 sample ready")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_mex():
    print(f"\n{'='*60}")
    print("  [6/6] MEx Multi-Modal Exercise (UCI)")
    print(f"{'='*60}")
    dest = os.path.join(RAW, "mex")
    os.makedirs(dest, exist_ok=True)
    if is_populated(dest):
        print("  Already downloaded — skipping")
        return True
    url = "https://archive.ics.uci.edu/static/public/500/mex.zip"
    zip_path = os.path.join(RAW, "mex.zip")
    try:
        download_file(url, zip_path)
        if os.path.getsize(zip_path) < 1000:
            print("  UCI returned a redirect page, trying ucimlrepo...")
            os.remove(zip_path)
            return download_mex_ucimlrepo(dest)
        extract_zip(zip_path, dest)
        os.remove(zip_path)
        print("  ✓ MEx ready")
        return True
    except (zipfile.BadZipFile, Exception) as e:
        print(f"  UCI zip failed ({e}), trying ucimlrepo...")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return download_mex_ucimlrepo(dest)


def download_mex_ucimlrepo(dest):
    try:
        os.system(f"{sys.executable} -m pip install ucimlrepo -q")
        from ucimlrepo import fetch_ucirepo
        print("  Fetching via ucimlrepo (id=500)...")
        mex = fetch_ucirepo(id=500)
        import pandas as pd
        X = mex.data.features
        y = mex.data.targets
        X.to_csv(os.path.join(dest, "mex_features.csv"), index=False)
        if y is not None:
            y.to_csv(os.path.join(dest, "mex_targets.csv"), index=False)
        print(f"  Saved {len(X)} samples to CSV")
        print("  ✓ MEx ready (via ucimlrepo)")
        return True
    except Exception as e2:
        print(f"  ✗ ucimlrepo also failed: {e2}")
        print("  Please download manually from: https://data.mendeley.com/datasets/p89fwbzmkd/2")
        return False


def check_status():
    print(f"\n{'='*60}")
    print("  DATASET STATUS")
    print(f"{'='*60}")
    datasets = ["mhealth", "pamap2", "ppg_4week", "stroke_rehab", "capture24", "mex"]
    all_ok = True
    for name in datasets:
        dest = os.path.join(RAW, name)
        if is_populated(dest):
            n = sum(1 for _, _, files in os.walk(dest) for _ in files)
            print(f"  ✓ {name:15s}  {n:4d} files")
        else:
            print(f"  ✗ {name:15s}  MISSING")
            all_ok = False
    if all_ok:
        print("\n  All datasets ready! Next step: python preprocess_all.py")
    else:
        print("\n  Some datasets missing — check errors above.")
    return all_ok


if __name__ == "__main__":
    print("HUA-DTIP — Dataset Downloader")
    print("=" * 60)
    print(f"Download directory: {RAW}")

    results = {}
    results["mhealth"] = download_mhealth()
    results["pamap2"] = download_pamap2()
    results["ppg_4week"] = download_ppg_4week()
    results["stroke_rehab"] = download_stroke_rehab()
    results["capture24"] = download_capture24()
    results["mex"] = download_mex()

    check_status()
