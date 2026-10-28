
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "Darknet.CSV"

TARGET = "Label.1"
TIMESTAMP = "Timestamp"
CUTOFF = pd.Timestamp("2016-02-24")

EXCLUDED_COLUMNS = {
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
    "Label",
    "Label.1",
}

df = pd.read_csv(DATA_PATH, low_memory=False)

df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()

df[TIMESTAMP] = pd.to_datetime(
    df[TIMESTAMP],
    format="%d/%m/%Y %I:%M:%S %p",
    errors="coerce",
)

df = df.dropna(subset=[TIMESTAMP])
df = df[df[TARGET] == "file-transfer"].copy()

df["period"] = np.where(
    df[TIMESTAMP] < CUTOFF,
    "Training",
    "Test",
)

features = [
    col for col in df.columns
    if col not in EXCLUDED_COLUMNS | {"period"}
    and pd.api.types.is_numeric_dtype(df[col])
]

rows = []

for feature in features:
    train = pd.to_numeric(
        df.loc[df["period"] == "Training", feature],
        errors="coerce",
    ).to_numpy(dtype=float)

    test = pd.to_numeric(
        df.loc[df["period"] == "Test", feature],
        errors="coerce",
    ).to_numpy(dtype=float)

    train = train[~np.isnan(train)]
    test = test[~np.isnan(test)]

    if len(train) == 0 or len(test) == 0:
        continue

    train_finite = train[np.isfinite(train)]
    test_finite = test[np.isfinite(test)]

    if len(train_finite) == 0 or len(test_finite) == 0:
        continue

    rows.append({
        "feature": feature,
        "train_min": np.min(train_finite),
        "train_max": np.max(train_finite),
        "test_min": np.min(test_finite),
        "test_max": np.max(test_finite),
        "train_nonfinite": len(train) - len(train_finite),
        "test_nonfinite": len(test) - len(test_finite),
        "train_zero_pct": 100 * np.mean(train_finite == 0),
        "test_zero_pct": 100 * np.mean(test_finite == 0),
        "train_missing_pct": 100 * (
            1 - len(train) / len(
                df.loc[df["period"] == "Training", feature]
            )
        ),
        "test_missing_pct": 100 * (
            1 - len(test) / len(
                df.loc[df["period"] == "Test", feature]
            )
        ),
    })

result = pd.DataFrame(rows)

print("=" * 70)
print("FILE-TRANSFER EXTREME-VALUE INSPECTION")
print("=" * 70)

print("\nLargest test-period maximum values:")
print(
    result.sort_values("test_max", ascending=False)
    .head(15)
    .to_string(index=False)
)

print("\nFeatures with the greatest difference in zero percentages:")
result["zero_percentage_difference"] = abs(
    result["test_zero_pct"] - result["train_zero_pct"]
)

print(
    result.sort_values(
        "zero_percentage_difference",
        ascending=False,
    )
    .head(15)
    .to_string(index=False)
)

print("\nInspection complete. No dataset or model files were changed.")
