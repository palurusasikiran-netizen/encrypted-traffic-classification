
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/Darknet.CSV")
CUTOFF = pd.Timestamp("2016-02-24")

print("=" * 65)
print("DATASET SPLIT DIAGNOSTIC")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)

print("\n1. Original dataset")
print("Total rows:", len(df))
print("Columns:", len(df.columns))

print("\n2. Label columns")
for col in ["Label", "Label.1"]:
    if col in df.columns:
        print(f"\n{col} class counts:")
        print(df[col].astype(str).str.strip().value_counts().to_string())
    else:
        print(f"{col}: NOT FOUND")

if "Timestamp" not in df.columns:
    raise ValueError("Timestamp column is missing.")

print("\n3. Timestamp parsing")

timestamp_formats = [
    "%d/%m/%Y %I:%M:%S %p",
    None
]

for fmt in timestamp_formats:
    if fmt is None:
        parsed = pd.to_datetime(df["Timestamp"], errors="coerce")
        format_name = "Automatic parsing"
    else:
        parsed = pd.to_datetime(
            df["Timestamp"],
            format=fmt,
            errors="coerce"
        )
        format_name = fmt

    print(f"\nFormat: {format_name}")
    print("Successfully parsed:", int(parsed.notna().sum()))
    print("Failed to parse:", int(parsed.isna().sum()))

    valid = parsed.notna()
    train_mask = valid & (parsed < CUTOFF)
    test_mask = valid & (parsed >= CUTOFF)

    print("Training rows:", int(train_mask.sum()))
    print("Testing rows:", int(test_mask.sum()))

    if "Label.1" in df.columns:
        labels = df["Label.1"].astype(str).str.strip().str.lower()

        print("\nTraining class counts:")
        print(labels[train_mask].value_counts().to_string())

        print("\nTesting class counts:")
        print(labels[test_mask].value_counts().to_string())

print("\n4. Dataset diagnostic completed.")
print("No files were modified.")
