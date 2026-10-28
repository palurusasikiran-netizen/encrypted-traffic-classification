
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "Darknet.CSV"

df = pd.read_csv(DATA_PATH, low_memory=False)

df["Label.1"] = (
    df["Label.1"].astype(str).str.strip().str.lower()
)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d/%m/%Y %I:%M:%S %p",
    errors="coerce",
)

df = df[
    (df["Label.1"] == "file-transfer")
    & df["Timestamp"].notna()
].copy()

df["period"] = np.where(
    df["Timestamp"] < pd.Timestamp("2016-02-24"),
    "Training",
    "Test",
)

columns = [
    "Timestamp",
    "Flow Duration",
    "Idle Min",
    "Idle Mean",
    "Idle Max",
    "Idle Std",
    "Dst Port",
    "Down/Up Ratio",
    "FIN Flag Count",
]

columns = [col for col in columns if col in df.columns]

for col in columns:
    if col != "Timestamp":
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("=" * 70)
print("RAW FILE-TRANSFER RECORD VERIFICATION")
print("=" * 70)

print("\nRepresentative training records with the largest Idle Min:")
print(
    df[df["period"] == "Training"]
    .nlargest(5, "Idle Min")[columns]
    .to_string(index=False)
)

print("\nRepresentative test records with the largest Idle Min:")
print(
    df[df["period"] == "Test"]
    .nlargest(5, "Idle Min")[columns]
    .to_string(index=False)
)

print("\nTest records with Idle Min greater than 1e14:")
print(
    df[
        (df["period"] == "Test")
        & (df["Idle Min"] > 1e14)
    ][columns]
    .head(10)
    .to_string(index=False)
)

print("\nRaw-record inspection complete. No files were changed.")
