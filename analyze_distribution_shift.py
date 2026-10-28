
import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# ============================================================
# 1. Configuration
# ============================================================

DATA_PATH = "data/Darknet.CSV"
RESULTS_DIR = "results"
CUTOFF = pd.Timestamp("2016-02-24")

os.makedirs(RESULTS_DIR, exist_ok=True)

CLASSES = [
    "audio-streaming",
    "browsing",
    "file-transfer",
    "p2p"
]

# ============================================================
# 2. Load the dataset and reproduce the temporal split
# ============================================================

print("=" * 70)
print("ISSUE 2: NETWORK DISTRIBUTION SHIFT ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_PATH, low_memory=False)

df["Label.1"] = (
    df["Label.1"].astype(str).str.strip().str.lower()
)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d/%m/%Y %I:%M:%S %p",
    errors="coerce"
)

df = df.dropna(subset=["Timestamp", "Label.1"]).copy()

train_df = df[df["Timestamp"] < CUTOFF].copy()
test_df = df[df["Timestamp"] >= CUTOFF].copy()

print("\nTotal dataset rows:", len(df))
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

# ============================================================
# 3. Analyze class distribution shift
# ============================================================

print("\n" + "=" * 70)
print("A. CLASS DISTRIBUTION SHIFT")
print("=" * 70)

train_counts = train_df["Label.1"].value_counts()
test_counts = test_df["Label.1"].value_counts()

class_table = pd.DataFrame({
    "train_count": train_counts,
    "test_count": test_counts
}).fillna(0).astype(int)

class_table["train_percentage"] = (
    class_table["train_count"] / len(train_df) * 100
)

class_table["test_percentage"] = (
    class_table["test_count"] / len(test_df) * 100
)

class_table["percentage_point_change"] = (
    class_table["test_percentage"]
    - class_table["train_percentage"]
)

class_table = class_table.sort_values(
    "test_count", ascending=False
)

print("\nClass proportions in training vs testing:")
print(class_table.round(3).to_string())

class_table.to_csv(
    os.path.join(RESULTS_DIR, "shift_class_distribution.csv")
)

# ============================================================
# 4. Analyze numeric feature distribution shift
# ============================================================

print("\n" + "=" * 70)
print("B. NUMERIC FEATURE DISTRIBUTION SHIFT")
print("=" * 70)

excluded_columns = {
    "Label",
    "Label.1",
    "Timestamp",
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Source IP",
    "Destination IP"
}

feature_columns = [
    col for col in df.columns
    if col not in excluded_columns
    and pd.api.types.is_numeric_dtype(df[col])
]

X_train = train_df[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)

X_test = test_df[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)

# Convert to numeric arrays and impute using training medians.
available = X_train.notna().any(axis=0)

X_train = X_train.loc[:, available]
X_test = X_test.loc[:, available]

imputer = SimpleImputer(strategy="median")

X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

# Remove features that are constant in training.
keep = np.var(X_train_imp, axis=0) > 0

kept_columns = X_train.columns[keep].tolist()

X_train_imp = X_train_imp[:, keep]
X_test_imp = X_test_imp[:, keep]

feature_rows = []

for i, feature in enumerate(kept_columns):

    train_values = X_train_imp[:, i]
    test_values = X_test_imp[:, i]

    train_median = np.median(train_values)
    test_median = np.median(test_values)

    train_q25 = np.percentile(train_values, 25)
    train_q75 = np.percentile(train_values, 75)

    test_q25 = np.percentile(test_values, 25)
    test_q75 = np.percentile(test_values, 75)

    train_std = np.std(train_values)
    test_std = np.std(test_values)

    # Difference between medians scaled by training standard deviation.
    # This is a descriptive shift measure, not a formal hypothesis test.
    if train_std > 0:
        standardized_median_shift = (
            (test_median - train_median) / train_std
        )
    else:
        standardized_median_shift = np.nan

    feature_rows.append({
        "feature": feature,
        "train_median": train_median,
        "test_median": test_median,
        "train_q25": train_q25,
        "train_q75": train_q75,
        "test_q25": test_q25,
        "test_q75": test_q75,
        "train_std": train_std,
        "test_std": test_std,
        "standardized_median_shift": standardized_median_shift,
        "absolute_shift": abs(standardized_median_shift)
        if np.isfinite(standardized_median_shift)
        else np.nan
    })

feature_shift = pd.DataFrame(feature_rows)

feature_shift = feature_shift.sort_values(
    "absolute_shift", ascending=False
)

print("\nNumber of numeric features before preprocessing:",
      len(feature_columns))
print("Number of usable numeric features:", len(kept_columns))

print("\nTop 20 features by standardized median shift:")
print(feature_shift.head(20).round(4).to_string(index=False))

feature_shift.to_csv(
    os.path.join(RESULTS_DIR, "shift_numeric_features.csv"),
    index=False
)

# ============================================================
# 5. Compare class proportions using total variation distance
# ============================================================

print("\n" + "=" * 70)
print("C. OVERALL CLASS-PROPORTION SHIFT")
print("=" * 70)

all_classes = sorted(
    set(train_df["Label.1"]) | set(test_df["Label.1"])
)

p_train = (
    train_df["Label.1"].value_counts(normalize=True)
    .reindex(all_classes, fill_value=0)
)

p_test = (
    test_df["Label.1"].value_counts(normalize=True)
    .reindex(all_classes, fill_value=0)
)

# Total variation distance ranges from 0 to 1.
# 0 means identical class proportions.
# Larger values mean greater differences.
tv_distance = 0.5 * np.abs(p_train - p_test).sum()

print(f"Total variation distance: {tv_distance:.4f}")
print("Interpretation: 0 means identical class proportions;")
print("larger values indicate a greater class-proportion difference.")

# ============================================================
# 6. Save summary
# ============================================================

summary_path = os.path.join(
    RESULTS_DIR, "distribution_shift_summary.txt"
)

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("NETWORK DISTRIBUTION SHIFT ANALYSIS\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Dataset rows: {len(df)}\n")
    file.write(f"Training rows: {len(train_df)}\n")
    file.write(f"Testing rows: {len(test_df)}\n")
    file.write(f"Temporal cutoff: {CUTOFF}\n\n")

    file.write("Class distribution:\n")
    file.write(class_table.round(3).to_string())
    file.write("\n\n")

    file.write(
        f"Class-proportion total variation distance: "
        f"{tv_distance:.6f}\n\n"
    )

    file.write("Top 20 numeric features by standardized median shift:\n")
    file.write(feature_shift.head(20).round(4).to_string(index=False))
    file.write("\n")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print("\nNew files saved:")
print("1. results/shift_class_distribution.csv")
print("2. results/shift_numeric_features.csv")
print("3. results/distribution_shift_summary.txt")

print("\nExisting model results were not modified.")
