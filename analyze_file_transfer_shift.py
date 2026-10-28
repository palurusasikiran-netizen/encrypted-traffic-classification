
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "Darknet.CSV"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "file_transfer_feature_shift.csv"
TARGET = "Label.1"
TIMESTAMP = "Timestamp"
CUTOFF = pd.Timestamp("2016-02-24")

EXCLUDED_COLUMNS = [
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
    "Label",
    "Label.1",
]

# --------------------------------------------------
# Load and prepare data
# --------------------------------------------------

print("=" * 65)
print("FILE-TRANSFER FEATURE DISTRIBUTION ANALYSIS")
print("=" * 65)

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

train_df = df[df["period"] == "Training"]
test_df = df[df["period"] == "Test"]

print(f"\nTraining file-transfer rows: {len(train_df):,}")
print(f"Test file-transfer rows:     {len(test_df):,}")

if train_df.empty or test_df.empty:
    raise ValueError(
        "Both training and test file-transfer samples are required."
    )

# --------------------------------------------------
# Select numeric features
# --------------------------------------------------

feature_columns = [
    column
    for column in df.columns
    if column not in EXCLUDED_COLUMNS + ["period"]
    and pd.api.types.is_numeric_dtype(df[column])
]

print(f"Numeric features available: {len(feature_columns)}")

# --------------------------------------------------
# Compare feature distributions
# --------------------------------------------------

results = []

for feature in feature_columns:
    train_values = pd.to_numeric(
        train_df[feature], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()

    test_values = pd.to_numeric(
        test_df[feature], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if len(train_values) < 2 or len(test_values) < 2:
        continue

    train_median = train_values.median()
    test_median = test_values.median()

    train_q25 = train_values.quantile(0.25)
    train_q75 = train_values.quantile(0.75)
    test_q25 = test_values.quantile(0.25)
    test_q75 = test_values.quantile(0.75)

    # Difference in medians relative to the combined
    # interquartile range. This is a descriptive measure,
    # not a formal statistical significance test.
    combined_iqr = max(
        train_q75 - train_q25,
        test_q75 - test_q25,
    )

    if combined_iqr > 0:
        scaled_median_difference = (
            abs(test_median - train_median) / combined_iqr
        )
    else:
        scaled_median_difference = (
            0.0 if train_median == test_median else np.inf
        )

    results.append(
        {
            "feature": feature,
            "train_valid_count": len(train_values),
            "test_valid_count": len(test_values),
            "train_median": train_median,
            "test_median": test_median,
            "train_q25": train_q25,
            "train_q75": train_q75,
            "test_q25": test_q25,
            "test_q75": test_q75,
            "absolute_median_difference": abs(
                test_median - train_median
            ),
            "scaled_median_difference": scaled_median_difference,
        }
    )

results_df = pd.DataFrame(results)

if results_df.empty:
    raise ValueError("No features had enough valid values to compare.")

results_df = results_df.sort_values(
    "scaled_median_difference",
    ascending=False,
)

results_df.to_csv(OUTPUT_CSV, index=False)

print("\nTOP 15 FEATURES WITH THE LARGEST DISTRIBUTION DIFFERENCES")
print("-" * 65)

display_columns = [
    "feature",
    "train_median",
    "test_median",
    "scaled_median_difference",
]

print(
    results_df[display_columns]
    .head(15)
    .to_string(index=False)
)

print("\nSaved full results to:")
print(OUTPUT_CSV)

print(
    "\nNote: large differences suggest potential distribution shift. "
    "They do not, by themselves, prove that a feature caused "
    "the model's classification errors."
)
