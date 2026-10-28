
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_FILE = Path("data") / "Darknet.CSV"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_FILE, low_memory=False)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d/%m/%Y %I:%M:%S %p",
    errors="coerce",
)
df["Label.1"] = (
    df["Label.1"].astype("string").str.strip().str.lower()
)

df = df.dropna(subset=["Timestamp", "Label.1"]).copy()

train_mask = df["Timestamp"] < pd.Timestamp("2016-02-24")
test_mask = ~train_mask

common_classes = sorted(
    set(df.loc[train_mask, "Label.1"])
    & set(df.loc[test_mask, "Label.1"])
)

df = df[df["Label.1"].isin(common_classes)].copy()
train_mask = df["Timestamp"] < pd.Timestamp("2016-02-24")
test_mask = ~train_mask

excluded = {
    "Flow ID", "Src IP", "Dst IP", "Timestamp", "Label", "Label.1"
}

features = [
    c for c in df.columns
    if c not in excluded
    and pd.api.types.is_numeric_dtype(df[c])
]

X_train_raw = df.loc[train_mask, features].copy()
X_test_raw = df.loc[test_mask, features].copy()

y_train = df.loc[train_mask, "Label.1"]
y_test = df.loc[test_mask, "Label.1"]

X_train_raw = X_train_raw.replace([np.inf, -np.inf], np.nan)
X_test_raw = X_test_raw.replace([np.inf, -np.inf], np.nan)

valid_features = X_train_raw.columns[
    X_train_raw.notna().any(axis=0)
].tolist()

X_train_raw = X_train_raw[valid_features]
X_test_raw = X_test_raw[valid_features]

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train_raw)
X_test = imputer.transform(X_test_raw)

nonconstant = np.std(X_train, axis=0) > 0
X_train = X_train[:, nonconstant]
X_test = X_test[:, nonconstant]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Features:", X_train.shape[1])
print("Common classes:", common_classes)

model = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=42,
)

print("\nTraining temporal baseline...")
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test, predictions, average="macro", zero_division=0
)

report = classification_report(
    y_test,
    predictions,
    labels=common_classes,
    zero_division=0,
)

print("\nTEMPORAL BASELINE RESULTS")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")
print(report)

with open(
    RESULTS_DIR / "fair_temporal_baseline.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write("TEMPORAL BASELINE\n")
    f.write("Training: before 2016-02-24\n")
    f.write("Testing: 2016-02-24 onward\n")
    f.write(f"Training rows: {len(X_train)}\n")
    f.write(f"Testing rows: {len(X_test)}\n")
    f.write(f"Features: {X_train.shape[1]}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro-F1: {macro_f1:.4f}\n\n")
    f.write(report)

print("\nSaved results/fair_temporal_baseline.txt")
