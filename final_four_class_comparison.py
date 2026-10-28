
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report
)

# ============================================================
# 1. Configuration
# ============================================================

SEED = 42
CUTOFF = pd.Timestamp("2016-02-24")

DATA_PATH = "data/Darknet.CSV"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CLASSES = [
    "audio-streaming",
    "browsing",
    "file-transfer",
    "p2p"
]

IDLE_FEATURES = [
    "Idle Min",
    "Idle Mean",
    "Idle Max",
    "Idle Std"
]

# ============================================================
# 2. Load data and reproduce the temporal split
# ============================================================

print("=" * 70)
print("FINAL FOUR-CLASS CONTROLLED COMPARISON")
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

# Keep only the four classes present in the temporal test set.
train_df = train_df[train_df["Label.1"].isin(CLASSES)].copy()
test_df = test_df[test_df["Label.1"].isin(CLASSES)].copy()

print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))

print("\nTraining class counts:")
print(train_df["Label.1"].value_counts().reindex(CLASSES, fill_value=0))

print("\nTesting class counts:")
print(test_df["Label.1"].value_counts().reindex(CLASSES, fill_value=0))

if set(train_df["Label.1"].unique()) != set(CLASSES):
    raise ValueError("Training data does not contain all four classes.")

if set(test_df["Label.1"].unique()) != set(CLASSES):
    raise ValueError("Test data does not contain all four classes.")

# ============================================================
# 3. Select numeric features
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

feature_columns = [
    col for col in df.columns
    if col not in excluded_columns
    and pd.api.types.is_numeric_dtype(df[col])
]

# Convert values safely and handle infinite values as missing.
X_train_raw = train_df[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)

X_test_raw = test_df[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)

y_train = train_df["Label.1"].to_numpy()
y_test = test_df["Label.1"].to_numpy()

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("\nInitial numeric features:", len(feature_columns))
print("Classes:", list(label_encoder.classes_))

# ============================================================
# 4. Train-only preprocessing
# ============================================================

def preprocess(X_train_input, X_test_input):

    # Remove features entirely missing in training.
    available = X_train_input.notna().any(axis=0)

    X_train_input = X_train_input.loc[:, available]
    X_test_input = X_test_input.loc[:, available]

    # Fit imputer on training data only.
    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train_input)
    X_test = imputer.transform(X_test_input)

    # Remove constant training features.
    keep = np.var(X_train, axis=0) > 0

    X_train = X_train[:, keep]
    X_test = X_test[:, keep]

    # Signed logarithmic transformation.
    X_train = np.sign(X_train) * np.log1p(np.abs(X_train))
    X_test = np.sign(X_test) * np.log1p(np.abs(X_test))

    # Fit scaler on training data only.
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.clip(X_train, -10, 10).astype(np.float32)
    X_test = np.clip(X_test, -10, 10).astype(np.float32)

    return X_train, X_test


# ============================================================
# 5. Autoencoder
# ============================================================

class Autoencoder(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# ============================================================
# 6. Run one experiment
# ============================================================

def run_experiment(name, selected_features):

    print("\n" + "=" * 70)
    print("EXPERIMENT:", name)
    print("=" * 70)

    X_train, X_test = preprocess(
        X_train_raw[selected_features],
        X_test_raw[selected_features]
    )

    input_dim = X_train.shape[1]
    print("Features after preprocessing:", input_dim)

    torch.manual_seed(SEED)

    model = Autoencoder(input_dim)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )
    loss_fn = nn.MSELoss()

    train_tensor = torch.from_numpy(X_train)

    # Unsupervised autoencoder training.
    model.train()

    for epoch in range(10):
        optimizer.zero_grad()

        reconstruction = model(train_tensor)
        loss = loss_fn(reconstruction, train_tensor)

        loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch + 1:02d}/10 "
            f"- Loss: {loss.item():.6f}"
        )

    # Extract embeddings.
    model.eval()

    with torch.no_grad():
        train_embeddings = model.encoder(
            torch.from_numpy(X_train)
        ).numpy()

        test_embeddings = model.encoder(
            torch.from_numpy(X_test)
        ).numpy()

    # Supervised classifier.
    classifier = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=SEED
    )

    classifier.fit(train_embeddings, y_train_encoded)

    predictions = classifier.predict(test_embeddings)

    accuracy = accuracy_score(y_test_encoded, predictions)

    macro_f1 = f1_score(
        y_test_encoded,
        predictions,
        labels=np.arange(len(CLASSES)),
        average="macro",
        zero_division=0
    )

    precision, recall, f1_values, support = (
        precision_recall_fscore_support(
            y_test_encoded,
            predictions,
            labels=np.arange(len(CLASSES)),
            zero_division=0
        )
    )

    report = classification_report(
        y_test_encoded,
        predictions,
        labels=np.arange(len(CLASSES)),
        target_names=label_encoder.classes_,
        zero_division=0
    )

    print(f"\nAccuracy: {accuracy:.4%}")
    print(f"Macro-F1: {macro_f1:.4%}")
    print("\nClassification report:")
    print(report)

    # Save new experiment outputs.
    predictions_path = os.path.join(
        RESULTS_DIR, f"{name}_predictions.csv"
    )

    pd.DataFrame({
        "actual_label": y_test,
        "predicted_label": label_encoder.inverse_transform(predictions)
    }).to_csv(predictions_path, index=False)

    report_path = os.path.join(
        RESULTS_DIR, f"{name}_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"Experiment: {name}\n")
        file.write(f"Training rows: {len(train_df)}\n")
        file.write(f"Testing rows: {len(test_df)}\n")
        file.write(f"Features: {input_dim}\n")
        file.write(f"Accuracy: {accuracy:.6f}\n")
        file.write(f"Macro-F1: {macro_f1:.6f}\n\n")
        file.write(report)

    torch.save(
        model.state_dict(),
        os.path.join(
            RESULTS_DIR, f"{name}_autoencoder_weights.pt"
        )
    )

    result = {
        "experiment": name,
        "features": input_dim,
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }

    for i, class_name in enumerate(label_encoder.classes_):
        result[f"{class_name}_precision"] = precision[i]
        result[f"{class_name}_recall"] = recall[i]
        result[f"{class_name}_f1"] = f1_values[i]
        result[f"{class_name}_support"] = support[i]

    return result


# ============================================================
# 7. Compare all features vs. removing idle-time features
# ============================================================

all_result = run_experiment(
    "final_four_class_log_all",
    feature_columns
)

features_without_idle = [
    col for col in feature_columns
    if col not in IDLE_FEATURES
]

removed_features = [
    col for col in feature_columns
    if col in IDLE_FEATURES
]

print("\nIdle features removed:", removed_features)

if not removed_features:
    raise ValueError(
        "Idle features were not found. Check the dataset column names."
    )

without_idle_result = run_experiment(
    "final_four_class_log_without_idle",
    features_without_idle
)

comparison = pd.DataFrame([
    all_result,
    without_idle_result
])

comparison_path = os.path.join(
    RESULTS_DIR, "final_four_class_comparison.csv"
)

comparison.to_csv(comparison_path, index=False)

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    comparison[
        ["experiment", "features", "accuracy", "macro_f1"]
    ].to_string(index=False)
)

print("\nFILE-TRANSFER METRICS")

print(
    comparison[
        [
            "experiment",
            "file-transfer_precision",
            "file-transfer_recall",
            "file-transfer_f1",
            "file-transfer_support"
        ]
    ].to_string(index=False)
)

print("\nComparison saved to:", comparison_path)
print("Existing experiment files were not overwritten.")
