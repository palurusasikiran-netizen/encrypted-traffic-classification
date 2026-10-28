
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

DATA_FILE = Path("data") / "Darknet.CSV"
RANDOM_STATE = 42

print("Loading dataset...")
df = pd.read_csv(DATA_FILE, low_memory=False)
print("Original shape:", df.shape)

# Normalize label spelling/capitalization.
df["Label.1"] = (
    df["Label.1"].astype("string").str.strip().str.lower()
)

print("\nNormalized application classes:")
print(df["Label.1"].value_counts())

# Remove identifiers and both label columns from input features.
# This avoids directly feeding IPs, flow IDs, timestamps, or labels
# into the classifier.
excluded = {
    "Flow ID", "Src IP", "Dst IP", "Timestamp",
    "Label", "Label.1",
}
feature_columns = [c for c in df.columns if c not in excluded]

# Keep numeric traffic features only.
X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# Remove columns that contain no usable values or no variation.
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=True) > 1]

# Remove rows with missing target labels.
y = df["Label.1"]
valid = y.notna() & y.ne("")
X = X.loc[valid].copy()
y = y.loc[valid].copy()

# Drop rows where every remaining feature is missing.
usable = X.notna().any(axis=1)
X = X.loc[usable]
y = y.loc[usable]

# Replace missing numeric feature values with each column's median.
medians = X.median()
X = X.fillna(medians)

# Keep the split reproducible and preserve class proportions.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Input features:", X.shape[1])

model = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=RANDOM_STATE,
)

print("\nTraining baseline model...")
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test, predictions, average="macro", zero_division=0
)

print("\n" + "=" * 55)
print("BASELINE RESULTS — RANDOM STRATIFIED SPLIT")
print("=" * 55)
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")
print("\nClassification report:")
print(
    classification_report(
        y_test, predictions, zero_division=0
    )
)

# Save the report and predictions for later comparison.
Path("results").mkdir(exist_ok=True)

report = classification_report(
    y_test, predictions, output_dict=True, zero_division=0
)
pd.DataFrame(report).transpose().to_csv(
    "results/baseline_report.csv"
)

pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": predictions,
}).to_csv("results/baseline_predictions.csv", index=False)

print("\nSaved results to the results/ folder.")
print(
    "Note: this random split is only the initial baseline; "
    "it does not establish robustness to distribution shifts."
)
