
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_FILE = Path("data") / "Darknet.CSV"
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_FILE, low_memory=False)

# Parse timestamps so we can test on later traffic.
df["_time"] = pd.to_datetime(
    df["Timestamp"], errors="coerce", dayfirst=True
)

df["Label.1"] = df["Label.1"].astype("string").str.strip().str.lower()
df = df.dropna(subset=["_time", "Label.1"])
df = df.sort_values("_time").reset_index(drop=True)

excluded = {
    "Flow ID", "Src IP", "Dst IP", "Timestamp",
    "Label", "Label.1", "_time",
}

feature_cols = [c for c in df.columns if c not in excluded]
X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# Remove completely missing and constant features.
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=True) > 1]
y = df["Label.1"].reset_index(drop=True)
X = X.reset_index(drop=True)

# Split chronologically: first 80% for training, last 20% for testing.
cut = int(len(df) * 0.8)
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = y.iloc[:cut], y.iloc[cut:]

print("\nTraining time range:")
print(df["_time"].iloc[0], "to", df["_time"].iloc[cut - 1])
print("Testing time range:")
print(df["_time"].iloc[cut], "to", df["_time"].iloc[-1])

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))
print("\nTraining class counts:")
print(y_train.value_counts().to_string())
print("\nTesting class counts:")
print(y_test.value_counts().to_string())

# Fit imputation only on training data to prevent preprocessing leakage.
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

model = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=42,
)

print("\nTraining on earlier traffic...")
model.fit(X_train, y_train)
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
macro_f1 = f1_score(
    y_test, pred, average="macro", zero_division=0
)

print("\n" + "=" * 55)
print("CHRONOLOGICAL EVALUATION")
print("=" * 55)
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")
print("\nClassification report:")
print(classification_report(y_test, pred, zero_division=0))

with open(RESULT_DIR / "temporal_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro-F1: {macro_f1:.4f}\n\n")
    f.write(classification_report(y_test, pred, zero_division=0))

print("\nSaved results/temporal_report.txt")
