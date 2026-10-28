
import os
import numpy as np
import pandas as pd

# ============================================================
# 1. Configuration
# ============================================================

DATA_PATH = "data/Darknet.CSV"
RESULTS_DIR = "results"
CUTOFF = pd.Timestamp("2016-02-24")

CLASSES = [
    "audio-streaming",
    "browsing",
    "file-transfer",
    "p2p"
]

IDLE_FEATURES = [
    "Idle Mean",
    "Idle Max",
    "Idle Min",
    "Idle Std"
]

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 2. Load and split dataset
# ============================================================

print("=" * 70)
print("WITHIN-CLASS FEATURE DISTRIBUTION SHIFT")
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

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

# ============================================================
# 3. Identify numeric network features
# ============================================================

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

features = [
    col for col in df.columns
    if col not in excluded_columns
    and pd.api.types.is_numeric_dtype(df[col])
]

# ============================================================
# 4. Compare features within each shared class
# ============================================================

results = []

for class_name in CLASSES:

    train_class = train_df[
        train_df["Label.1"] == class_name
    ]

    test_class = test_df[
        test_df["Label.1"] == class_name
    ]

    print("\n" + "-" * 70)
    print("CLASS:", class_name)
    print("Training samples:", len(train_class))
    print("Testing samples:", len(test_class))

    if train_class.empty or test_class.empty:
        print("Skipped: class is absent in one period.")
        continue

    for feature in features:

        train_values = pd.to_numeric(
            train_class[feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        test_values = pd.to_numeric(
            test_class[feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if len(train_values) == 0 or len(test_values) == 0:
            continue

        train_median = train_values.median()
        test_median = test_values.median()

        train_std = train_values.std()

        if pd.isna(train_std) or train_std == 0:
            shift = np.nan
        else:
            shift = (test_median - train_median) / train_std

        results.append({
            "class": class_name,
            "feature": feature,
            "train_count": len(train_values),
            "test_count": len(test_values),
            "train_median": train_median,
            "test_median": test_median,
            "train_q25": train_values.quantile(0.25),
            "train_q75": train_values.quantile(0.75),
            "test_q25": test_values.quantile(0.25),
            "test_q75": test_values.quantile(0.75),
            "standardized_median_shift": shift,
            "absolute_shift": abs(shift)
            if pd.notna(shift) else np.nan
        })

if not results:
    raise ValueError("No within-class comparisons could be performed.")

result_df = pd.DataFrame(results)

# Save every feature comparison separately.
output_path = os.path.join(
    RESULTS_DIR,
    "within_class_feature_shift.csv"
)

result_df.to_csv(output_path, index=False)

# ============================================================
# 5. Print the largest shifts for each class
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 SHIFTS WITHIN EACH CLASS")
print("=" * 70)

for class_name in CLASSES:

    subset = result_df[
        result_df["class"] == class_name
    ].dropna(subset=["absolute_shift"])

    subset = subset.sort_values(
        "absolute_shift", ascending=False
    ).head(10)

    print("\nCLASS:", class_name)

    if subset.empty:
        print("No valid feature shifts available.")
        continue

    print(
        subset[
            [
                "feature",
                "train_median",
                "test_median",
                "standardized_median_shift"
            ]
        ].round(4).to_string(index=False)
    )

# ============================================================
# 6. Save a summary
# ============================================================

summary_path = os.path.join(
    RESULTS_DIR,
    "within_class_shift_summary.txt"
)

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("WITHIN-CLASS FEATURE SHIFT ANALYSIS\n")
    file.write("=" * 50 + "\n\n")

    for class_name in CLASSES:

        subset = result_df[
            result_df["class"] == class_name
        ].dropna(subset=["absolute_shift"])

        subset = subset.sort_values(
            "absolute_shift", ascending=False
        ).head(10)

        file.write(f"\nCLASS: {class_name}\n")
        file.write(subset.round(4).to_string(index=False))
        file.write("\n\n")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print("Detailed results:", output_path)
print("Summary:", summary_path)
print("No model files were modified.")
