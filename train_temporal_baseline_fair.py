
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

SEED = 42
DATA_PATH = "data/Darknet.CSV"

df = pd.read_csv(DATA_PATH, low_memory=False)

# Normalize labels
df["Label.1"] = df["Label.1"].astype(str).str.strip().str.lower()
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"], format="%d/%m/%Y %I:%M:%S %p", errors="coerce"
)

# Keep the four classes shared by both time periods
classes = ["audio-streaming", "browsing", "file-transfer", "p2p"]
df = df[df["Label.1"].isin(classes)].copy()
df = df.dropna(subset=["Timestamp"])

# Chronological train/test split
cutoff = pd.Timestamp("2016-02-24")
train = df[df["Timestamp"] < cutoff].copy()
test = df[df["Timestamp"] >= cutoff].copy()

excluded = [
    "Flow ID", "Src IP", "Dst IP", "Timestamp", "Label", "Label.1"
]
feature_cols = [
    c for c in df.columns
    if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
]

X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan)
X_test = test[feature_cols].replace([np.inf, -np.inf], np.nan)
y_train = train["Label.1"]
y_test = test["Label.1"]

# Fit preprocessing ONLY on training data
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Remove columns that are constant in training data
keep = np.std(X_train, axis=0) > 0
X_train = X_train[:, keep]
X_test = X_test[:, keep]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Use the same clipping approach as the proposed pipeline
X_train = np.clip(X_train, -10, 10)
X_test = np.clip(X_test, -10, 10)

# Supervised baseline on the original preprocessed features
model = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=SEED
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Features used:", X_train.shape[1])
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Macro-F1:", round(f1_score(y_test, pred, average="macro"), 4))
print("\nClassification Report:\n")
print(classification_report(y_test, pred, labels=classes, zero_division=0))

# Save predictions for a fair, row-aligned evaluation
fair_predictions = pd.DataFrame({
    "actual": y_test.reset_index(drop=True),
    "predicted": pd.Series(pred)
})

fair_predictions.to_csv(
    "results/fair_temporal_baseline_predictions.csv",
    index=False
)

print("\nSaved results/fair_temporal_baseline_predictions.csv")
