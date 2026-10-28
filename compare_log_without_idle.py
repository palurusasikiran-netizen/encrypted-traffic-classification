
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

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SEED = 42
CUTOFF = pd.Timestamp("2016-02-24")

DATA_PATH = "data/Darknet.CSV"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")

# These four features will be removed in the second experiment.
IDLE_FEATURES = [
    "Idle Min",
    "Idle Mean",
    "Idle Max",
    "Idle Std"
]

# --------------------------------------------------
# 2. Load and prepare the dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH, low_memory=False)

required_columns = ["Label.1", "Timestamp"]
missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

df["Label.1"] = (
    df["Label.1"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d/%m/%Y %I:%M:%S %p",
    errors="coerce"
)

df = df.dropna(subset=["Timestamp", "Label.1"]).copy()

# Same temporal split as the existing experiments.
train_df = df[df["Timestamp"] < CUTOFF].copy()
test_df = df[df["Timestamp"] >= CUTOFF].copy()

print(f"Training rows: {len(train_df)}")
print(f"Testing rows:  {len(test_df)}")

if train_df.empty or test_df.empty:
    raise ValueError("The temporal train/test split is empty.")

# --------------------------------------------------
# 3. Select numeric features
# --------------------------------------------------

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

# Exclude identifiers, labels, timestamps, and IP addresses.
feature_columns = []

for col in df.columns:
    if col in excluded_columns:
        continue

    if pd.api.types.is_numeric_dtype(df[col]):
        feature_columns.append(col)

if not feature_columns:
    raise ValueError("No numeric features were found.")

# Convert selected columns to numeric safely.
for col in feature_columns:
    train_df[col] = pd.to_numeric(
        train_df[col], errors="coerce"
    )
    test_df[col] = pd.to_numeric(
        test_df[col], errors="coerce"
    )

# Match the existing experiment's feature selection.
X_train_raw = train_df[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)
X_test_raw = test_df[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)

y_train = train_df["Label.1"].to_numpy()
y_test = test_df["Label.1"].to_numpy()

# Use the same class encoding for both experiments.
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

unknown_labels = set(y_test) - set(label_encoder.classes_)
if unknown_labels:
    raise ValueError(
        f"Test contains unseen classes: {unknown_labels}"
    )

y_test_encoded = label_encoder.transform(y_test)

print(f"Initial numeric features: {len(feature_columns)}")
print("Classes:", list(label_encoder.classes_))

# --------------------------------------------------
# 4. Shared preprocessing
# --------------------------------------------------

def preprocess_features(X_train_input, X_test_input):
    """
    Fit preprocessing on training data only.
    Apply the fitted transformations to the test data.
    """

    # Remove columns that are entirely missing in training.
    non_all_missing = X_train_input.notna().any(axis=0)

    X_train_input = X_train_input.loc[
        :, non_all_missing
    ].copy()

    X_test_input = X_test_input.loc[
        :, non_all_missing
    ].copy()

    # Imputation is fitted only on training data.
    imputer = SimpleImputer(strategy="median")

    X_train_imputed = imputer.fit_transform(X_train_input)
    X_test_imputed = imputer.transform(X_test_input)

    # Remove constant features based on training data.
    train_variances = np.var(X_train_imputed, axis=0)
    keep_features = train_variances > 0

    X_train_imputed = X_train_imputed[:, keep_features]
    X_test_imputed = X_test_imputed[:, keep_features]

    if X_train_imputed.shape[1] == 0:
        raise ValueError("No usable features remain after preprocessing.")

    # Signed log transformation.
    X_train_log = np.sign(X_train_imputed) * np.log1p(
        np.abs(X_train_imputed)
    )
    X_test_log = np.sign(X_test_imputed) * np.log1p(
        np.abs(X_test_imputed)
    )

    # Fit scaler on training data only.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_log)
    X_test_scaled = scaler.transform(X_test_log)

    # Match the existing experiment's clipping.
    X_train_scaled = np.clip(
        X_train_scaled, -10, 10
    ).astype(np.float32)

    X_test_scaled = np.clip(
        X_test_scaled, -10, 10
    ).astype(np.float32)

    return X_train_scaled, X_test_scaled


# --------------------------------------------------
# 5. Autoencoder definition
# --------------------------------------------------

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
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


# --------------------------------------------------
# 6. Train and evaluate one experiment
# --------------------------------------------------

def run_experiment(experiment_name, selected_features):

    print("\n" + "=" * 65)
    print("Experiment:", experiment_name)
    print("=" * 65)

    Xtr_raw = X_train_raw[selected_features]
    Xte_raw = X_test_raw[selected_features]

    Xtr, Xte = preprocess_features(Xtr_raw, Xte_raw)

    input_dim = Xtr.shape[1]

    print("Features used:", input_dim)

    # Reset seed so both experiments have the same
    # initialization and comparable training conditions.
    torch.manual_seed(SEED)

    model = Autoencoder(input_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )
    loss_function = nn.MSELoss()

    train_tensor = torch.from_numpy(Xtr).to(device)

    # Train the autoencoder without class labels.
    model.train()

    for epoch in range(10):
        optimizer.zero_grad()

        reconstructed = model(train_tensor)
        loss = loss_function(reconstructed, train_tensor)

        loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch + 1:02d}/10 "
            f"- Reconstruction loss: {loss.item():.6f}"
        )

    # Extract learned representations.
    model.eval()

    with torch.no_grad():
        train_embeddings = (
            model.encoder(torch.from_numpy(Xtr).to(device))
            .cpu().numpy()
        )

        test_embeddings = (
            model.encoder(torch.from_numpy(Xte).to(device))
            .cpu().numpy()
        )

    # Supervised classifier trained on learned embeddings.
    classifier = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=SEED
    )

    classifier.fit(train_embeddings, y_train_encoded)

    predictions_encoded = classifier.predict(test_embeddings)
    predictions = label_encoder.inverse_transform(
        predictions_encoded
    )

    accuracy = accuracy_score(y_test_encoded, predictions_encoded)
    macro_f1 = f1_score(
        y_test_encoded,
        predictions_encoded,
        average="macro",
        zero_division=0
    )

    precision, recall, f1_values, support = (
        precision_recall_fscore_support(
            y_test_encoded,
            predictions_encoded,
            labels=np.arange(len(label_encoder.classes_)),
            zero_division=0
        )
    )

    report = classification_report(
        y_test_encoded,
        predictions_encoded,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    )

    print(f"\nAccuracy: {accuracy:.4%}")
    print(f"Macro-F1: {macro_f1:.4%}")
    print("\nClassification report:")
    print(report)

    # Save separate files for this experiment.
    safe_name = (
        experiment_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    predictions_path = os.path.join(
        RESULTS_DIR,
        f"{safe_name}_predictions.csv"
    )

    output = pd.DataFrame({
        "actual_label": y_test,
        "predicted_label": predictions
    })

    output.to_csv(predictions_path, index=False)

    report_path = os.path.join(
        RESULTS_DIR,
        f"{safe_name}_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"Experiment: {experiment_name}\n")
        file.write(f"Training rows: {len(train_df)}\n")
        file.write(f"Testing rows: {len(test_df)}\n")
        file.write(f"Features used: {input_dim}\n")
        file.write(f"Accuracy: {accuracy:.6f}\n")
        file.write(f"Macro-F1: {macro_f1:.6f}\n\n")
        file.write(report)

    model_path = os.path.join(
        RESULTS_DIR,
        f"{safe_name}_autoencoder_weights.pt"
    )

    torch.save(model.state_dict(), model_path)

    results = {
        "experiment": experiment_name,
        "features": input_dim,
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }

    for i, class_name in enumerate(label_encoder.classes_):
        results[f"{class_name}_precision"] = precision[i]
        results[f"{class_name}_recall"] = recall[i]
        results[f"{class_name}_f1"] = f1_values[i]
        results[f"{class_name}_support"] = support[i]

    print("\nSaved predictions:", predictions_path)
    print("Saved report:", report_path)
    print("Saved model:", model_path)

    return results


# --------------------------------------------------
# 7. Run both controlled experiments
# --------------------------------------------------

print("\nRunning experiment 1: all numeric features")

all_features_result = run_experiment(
    "log_all_features",
    feature_columns
)

# Exclude the four specified idle-time features.
features_without_idle = [
    col for col in feature_columns
    if col not in IDLE_FEATURES
]

removed_features = [
    col for col in feature_columns
    if col in IDLE_FEATURES
]

print("\nIdle features found and removed:", removed_features)

if not removed_features:
    raise ValueError(
        "None of the specified idle features were found. "
        "Check their exact column names in Darknet.CSV."
    )

print("\nRunning experiment 2: without idle-time features")

without_idle_result = run_experiment(
    "log_without_idle",
    features_without_idle
)

# --------------------------------------------------
# 8. Save the comparison separately
# --------------------------------------------------

comparison = pd.DataFrame([
    all_features_result,
    without_idle_result
])

comparison_path = os.path.join(
    RESULTS_DIR,
    "log_idle_feature_comparison.csv"
)

comparison.to_csv(comparison_path, index=False)

print("\n" + "=" * 65)
print("FINAL COMPARISON")
print("=" * 65)

print(
    comparison[
        ["experiment", "features", "accuracy", "macro_f1"]
    ].to_string(index=False)
)

ft_columns = [
    col for col in comparison.columns
    if col.startswith("file-transfer_")
]

if ft_columns:
    print("\nFile-transfer comparison:")
    print(
        comparison[
            ["experiment"] + ft_columns
        ].to_string(index=False)
    )

print("\nComparison saved to:", comparison_path)
print("\nExperiment completed. Original result files were not overwritten.")
