"""
Dataset Acquisition Script for Smart Predictive Maintenance System
Downloads and extracts real datasets:
1. AI4I 2020 Predictive Maintenance Dataset (UCI)
2. NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset (FD001 to FD004)
"""

import os
import sys
import zipfile
import urllib.request
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI4I_DIR = os.path.join(BASE_DIR, "data", "raw", "ai4i")
CMAPSS_DIR = os.path.join(BASE_DIR, "data", "raw", "cmapss")

os.makedirs(AI4I_DIR, exist_ok=True)
os.makedirs(CMAPSS_DIR, exist_ok=True)

AI4I_URLS = [
    "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
    "https://raw.githubusercontent.com/marshald/ai4i2020/main/ai4i2020.csv",
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/stevedata/ai4i2020.csv"
]

CMAPSS_FILES = [
    "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
    "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
    "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
    "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt"
]

CMAPSS_BASE_URL = "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master/"

def download_file(url, dest_path):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=45) as response, open(dest_path, 'wb') as out_file:
            out_file.write(response.read())
        size_kb = os.path.getsize(dest_path) / 1024
        print(f"  Downloaded: {os.path.basename(dest_path)} ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        print(f"  Failed from {url}: {e}")
        return False

def acquire_ai4i():
    print("\n[1/2] Checking / Downloading AI4I 2020 Predictive Maintenance Dataset...")
    ai4i_csv = os.path.join(AI4I_DIR, "ai4i2020.csv")
    if os.path.exists(ai4i_csv) and os.path.getsize(ai4i_csv) > 10000:
        print(f"  AI4I dataset present: {ai4i_csv}")
    else:
        zip_path = os.path.join(AI4I_DIR, "ai4i.zip")
        download_file(AI4I_URLS[0], zip_path)
        if os.path.exists(zip_path) and zipfile.is_zipfile(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(AI4I_DIR)
            os.remove(zip_path)
        else:
            download_file(AI4I_URLS[1], ai4i_csv)

    if os.path.exists(ai4i_csv):
        df = pd.read_csv(ai4i_csv)
        print(f"  AI4I 2020 verification: shape={df.shape}, {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns)}")
        assert df.shape[0] >= 10000, f"Expected 10,000 rows, got {df.shape[0]}"
        print("  -> AI4I 2020 dataset OK.")
        return True
    else:
        print("  [ERROR] AI4I dataset could not be downloaded.")
        return False

def acquire_cmapss():
    print("\n[2/2] Checking / Downloading NASA C-MAPSS Turbofan Engine Dataset (FD001 - FD004)...")
    success_count = 0
    for filename in CMAPSS_FILES:
        dest_path = os.path.join(CMAPSS_DIR, filename)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100:
            print(f"  Already exists: {filename}")
            success_count += 1
            continue
        url = CMAPSS_BASE_URL + filename
        if download_file(url, dest_path):
            success_count += 1

    train_fd001 = os.path.join(CMAPSS_DIR, "train_FD001.txt")
    test_fd001 = os.path.join(CMAPSS_DIR, "test_FD001.txt")
    rul_fd001 = os.path.join(CMAPSS_DIR, "RUL_FD001.txt")

    if os.path.exists(train_fd001) and os.path.exists(test_fd001) and os.path.exists(rul_fd001):
        df_train = pd.read_csv(train_fd001, sep=r'\s+', header=None)
        df_test = pd.read_csv(test_fd001, sep=r'\s+', header=None)
        df_rul = pd.read_csv(rul_fd001, sep=r'\s+', header=None)
        print(f"  train_FD001 shape: {df_train.shape} (20,631 cycles across 100 engines)")
        print(f"  test_FD001 shape: {df_test.shape} (13,096 cycles across 100 test engines)")
        print(f"  RUL_FD001 shape: {df_rul.shape} (100 engine ground-truth RULs)")
        print("  -> NASA C-MAPSS dataset OK.")
        return True
    else:
        print("  [ERROR] NASA C-MAPSS FD001 files missing.")
        return False

if __name__ == "__main__":
    ok1 = acquire_ai4i()
    ok2 = acquire_cmapss()
    if ok1 and ok2:
        print("\nAll datasets acquired and verified successfully!")
        sys.exit(0)
    else:
        sys.exit(1)
