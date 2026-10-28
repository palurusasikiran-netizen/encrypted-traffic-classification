
import pandas as pd
from pathlib import Path

DATA_FILE = Path("data") / "Darknet.CSV"

print("=" * 60)
print("DARKNET DATASET INSPECTION")
print("=" * 60)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE.resolve()}")

# Read a small sample first
sample = pd.read_csv(DATA_FILE, nrows=5, low_memory=False)

print("\n1. COLUMN NAMES:")
for column in sample.columns:
    print(repr(column))

print("\n2. FIRST FIVE ROWS:")
print(sample.to_string(index=False))

print("\n3. DATASET SHAPE:")
rows = 0
for chunk in pd.read_csv(DATA_FILE, chunksize=50000, low_memory=False):
    rows += len(chunk)

print("Total rows:", rows)
print("Total columns:", len(sample.columns))

print("\n4. POSSIBLE LABEL COLUMNS:")
for column in sample.columns:
    if any(word in column.lower() for word in ["label", "class", "category", "traffic"]):
        print("-", repr(column))

print("\nInspection completed.")
