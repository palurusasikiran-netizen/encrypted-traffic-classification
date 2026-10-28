
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

features = [
    "Idle Min",
    "Idle Mean",
    "Idle Max",
    "Idle Std",
    "Flow Duration",
    "Dst Port",
    "Down/Up Ratio",
    "FIN Flag Count",
]

print("=" * 65)
print("FILE-TRANSFER RAW FEATURE INSPECTION")
print("=" * 65)

for feature in features:
    if feature not in df.columns:
        print(f"\n{feature}: not found")
        continue

    print(f"\n--- {feature} ---")

    for period in ["Training", "Test"]:
        values = pd.to_numeric(
            df.loc[df["period"] == period, feature],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if values.empty:
            print(f"{period}: no valid values")
            continue

        print(
            f"{period}: "
            f"count={len(values)}, "
            f"min={values.min():.6g}, "
            f"p25={values.quantile(0.25):.6g}, "
            f"median={values.median():.6g}, "
            f"p75={values.quantile(0.75):.6g}, "
            f"max={values.max():.6g}"
        )

print("\nInspection complete. No files were modified.")
