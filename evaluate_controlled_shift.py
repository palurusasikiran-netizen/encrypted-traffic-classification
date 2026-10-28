
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_FILE = Path("data") / "Darknet.CSV"
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE, low_memory=False)

df["_time"] = pd.to_datetime(
    df["Timestamp"],
    format="%d/%m/%Y %I:%M:%S %p",
    errors="coerce",
)
df["Label.1"] = (
    df["Label.1"].astype("string").str.strip().str.lower()
)
df = df.dropna(subset=["_time", "Label.1"]).copy()

# Use dates, not arbitrary row positions, to define the shift.
train_mask = df["_time"] < pd.Timestamp("2016-02-24")
test_mask = df["_time"] >= pd.Timestamp("2016-02-24")

train_labels = set(df.loc[train_mask, "Label.1"].unique())
test_labels = set(df.loc[test_mask, "Label.1"].unique())
common_labels = sorted(train_labels & test_labels)

print("Training dates: before 2016-02-24")
print("Testing dates:  2016-02-24 onward")
print("Classes in both periods:", common_labels)

train = df.loc[train_mask & df["Label.1"].isin(common_labels)].copy()
test = df.loc[test_mask & df["Label.1"].isin(common_labels)].copy()

excluded = {
    "Flow ID", "Src IP", "Dst IP", "Timestamp",
    "Label", "Label.1", "_time",
}
features = [c for c in df.columns if c not in excluded]

X_train = train[features].apply(pd.to_numeric, errors="coerce")
X_test = test[features].apply(pd.to_numeric, errors="coerce")

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Remove features that are entirely missing in training.
X_train = X_train.dropna(axis=1, how="all")
X_test = X_test[X_train.columns]

# Remove constant features using training data only.
varying = X_train.nunique(dropna=True) > 1
X_train = X_train.loc[:, varying]
X_test = X_test.loc[:, varying]

y_train = train["Label.1"]
y_test = test["Label.1"]

# Fit preprocessing only on training data.
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

model = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=42,
)

print("\nTraining rows:", len(y_train))
print("Testing rows:", len(y_test))
print("\nTraining distribution:")
print(y_train.value_counts().to_string())
print("\nTesting distribution:")
print(y_test.value_counts().to_string())

model.fit(X_train, y_train)
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)

print("\nCONTROLLED TEMPORAL-SHIFT BASELINE")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")
print(classification_report(y_test, pred, zero_division=0))

with open(RESULTS / "controlled_shift_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Classes: {common_labels}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro-F1: {macro_f1:.4f}\n\n")
    f.write(classification_report(y_test, pred, zero_division=0))

print("\nSaved results/controlled_shift_report.txt")
