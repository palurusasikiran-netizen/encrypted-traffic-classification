
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SEED = 42
EPOCHS = 10
BATCH_SIZE = 512
LATENT_DIM = 16

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "Darknet.CSV"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_REPORT = RESULTS_DIR / "file_transfer_threshold_report.txt"
OUTPUT_PREDICTIONS = (
    RESULTS_DIR / "file_transfer_threshold_predictions.csv"
)
OUTPUT_THRESHOLDS = (
    RESULTS_DIR / "file_transfer_threshold_validation.csv"
)
OUTPUT_WEIGHTS = (
    RESULTS_DIR / "file_transfer_threshold_autoencoder.pt"
)

TARGET = "Label.1"
TIMESTAMP = "Timestamp"
CUTOFF = pd.Timestamp("2016-02-24")
TARGET_CLASS = "file-transfer"

EXCLUDED_COLUMNS = [
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
    "Label",
    "Label.1",
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 65)
print("FILE-TRANSFER THRESHOLD-TUNING EXPERIMENT")
print("=" * 65)

# --------------------------------------------------
# 2. Load data and establish temporal test split
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)

if TARGET not in df.columns or TIMESTAMP not in df.columns:
    raise ValueError(
        f"Required columns missing: {TARGET} or {TIMESTAMP}"
    )

df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()

df[TIMESTAMP] = pd.to_datetime(
    df[TIMESTAMP],
    format="%d/%m/%Y %I:%M:%S %p",
    errors="coerce",
)

df = df.dropna(subset=[TARGET, TIMESTAMP]).copy()

train_mask = df[TIMESTAMP] < CUTOFF
test_mask = df[TIMESTAMP] >= CUTOFF

train_df = df.loc[train_mask].copy()
test_df = df.loc[test_mask].copy()

# Use the same four common classes as the earlier experiments.
common_classes = sorted(
    set(train_df[TARGET].unique())
    & set(test_df[TARGET].unique())
)

if TARGET_CLASS not in common_classes:
    raise ValueError(
        "file-transfer is not present in both temporal splits."
    )

train_df = train_df[
    train_df[TARGET].isin(common_classes)
].copy()

test_df = test_df[
    test_df[TARGET].isin(common_classes)
].copy()

train_df = train_df.sort_values(TIMESTAMP).reset_index(drop=True)
test_df = test_df.sort_values(TIMESTAMP).reset_index(drop=True)

print(f"\nTraining-period rows: {len(train_df):,}")
print(f"Final test rows:      {len(test_df):,}")
print(f"Classes: {common_classes}")

# --------------------------------------------------
# 3. Make a time-based validation split
# --------------------------------------------------

validation_start = train_df[TIMESTAMP].quantile(0.80)

inner_train_df = train_df[
    train_df[TIMESTAMP] < validation_start
].copy()

validation_df = train_df[
    train_df[TIMESTAMP] >= validation_start
].copy()

if inner_train_df.empty or validation_df.empty:
    raise ValueError("The internal validation split is empty.")

if not set(common_classes).issubset(
    set(inner_train_df[TARGET].unique())
):
    raise ValueError(
        "The earlier training portion does not contain every class. "
        "A valid time-based experiment needs a different split."
    )

if not set(common_classes).issubset(
    set(validation_df[TARGET].unique())
):
    raise ValueError(
        "The later validation portion does not contain every class. "
        "A valid threshold comparison cannot be made for all classes."
    )

print(f"\nValidation begins at: {validation_start}")
print(f"Inner training rows:   {len(inner_train_df):,}")
print(f"Validation rows:       {len(validation_df):,}")

print("\nValidation class counts:")
print(validation_df[TARGET].value_counts().to_string())

# --------------------------------------------------
# 4. Define the autoencoder
# --------------------------------------------------

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def make_features(frame, feature_columns):
    """Convert selected columns to numeric values."""
    return frame[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )


def fit_preprocessor(fit_frame, feature_columns):
    """Fit imputation and scaling only on the supplied training data."""
    X = make_features(fit_frame, feature_columns)

    # Remove columns that contain no usable values in this fit split.
    usable = X.columns[X.notna().any()].tolist()
    X = X[usable]

    if X.shape[1] == 0:
        raise ValueError("No usable numeric features were found.")

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Remove constant features after imputation.
    nonconstant = np.var(X_imputed, axis=0) > 0
    X_imputed = X_imputed[:, nonconstant]
    usable = [
        col for col, keep in zip(usable, nonconstant) if keep
    ]

    if len(usable) == 0:
        raise ValueError("No nonconstant features remain.")

    # Refit the imputer on the retained columns for a consistent pipeline.
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(
        make_features(fit_frame, usable)
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_scaled = np.clip(X_scaled, -10, 10).astype(np.float32)

    return usable, imputer, scaler, X_scaled


def transform_features(frame, feature_columns, imputer, scaler):
    X = make_features(frame, feature_columns)
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    return np.clip(X_scaled, -10, 10).astype(np.float32)


def train_autoencoder(X_train, epochs=10):
    """Learn an embedding using training features without labels."""
    torch.manual_seed(SEED)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)

    model = Autoencoder(
        input_dim=X_train.shape[1],
        latent_dim=LATENT_DIM,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    generator = torch.Generator()
    generator.manual_seed(SEED)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_rows = 0

        for (batch,) in loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)
            total_rows += len(batch)

        average_loss = total_loss / total_rows

        print(
            f"  Autoencoder epoch {epoch + 1}/{epochs}: "
            f"loss = {average_loss:.6f}"
        )

    model.eval()

    with torch.no_grad():
        embeddings = model.encoder(X_tensor).numpy()

    return model, embeddings


def encode_data(model, X):
    model.eval()
    with torch.no_grad():
        return model.encoder(
            torch.tensor(X, dtype=torch.float32)
        ).numpy()


# --------------------------------------------------
# 5. Select the feature columns
# --------------------------------------------------

feature_columns = [
    col for col in df.columns
    if col not in EXCLUDED_COLUMNS
]

feature_columns = [
    col for col in feature_columns
    if pd.api.types.is_numeric_dtype(df[col])
]

if not feature_columns:
    raise ValueError("No numeric feature columns found.")

y_inner = inner_train_df[TARGET].to_numpy()
y_validation = validation_df[TARGET].to_numpy()

# --------------------------------------------------
# 6. Validation experiment: fit on earlier data
# --------------------------------------------------

print("\n" + "=" * 65)
print("PART A: VALIDATION-BASED THRESHOLD SELECTION")
print("=" * 65)

val_features, val_imputer, val_scaler, X_inner = fit_preprocessor(
    inner_train_df, feature_columns
)

X_validation = transform_features(
    validation_df, val_features, val_imputer, val_scaler
)

print(f"\nFeatures retained: {len(val_features)}")
print("Training validation-stage autoencoder...")

val_autoencoder, Z_inner = train_autoencoder(
    X_inner, epochs=EPOCHS
)

Z_validation = encode_data(val_autoencoder, X_validation)

val_classifier = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=SEED,
)

val_classifier.fit(Z_inner, y_inner)

probabilities = val_classifier.predict_proba(Z_validation)
class_order = list(val_classifier.classes_)

if TARGET_CLASS not in class_order:
    raise ValueError("file-transfer is missing from classifier classes.")

target_index = class_order.index(TARGET_CLASS)
target_probabilities = probabilities[:, target_index]

# Evaluate candidate thresholds using validation labels only.
# Thresholds are chosen from observed validation probabilities.
candidate_thresholds = np.unique(
    np.concatenate(
        [
            np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20,
                      0.30, 0.40, 0.50]),
            target_probabilities,
        ]
    )
)

validation_results = []

for threshold in candidate_thresholds:
    predicted = np.array(class_order)[
        np.argmax(probabilities, axis=1)
    ].astype(object)

    # If file-transfer probability reaches the threshold, assign it.
    predicted[target_probabilities >= threshold] = TARGET_CLASS

    precision = precision_score(
        y_validation,
        predicted,
        labels=[TARGET_CLASS],
        average=None,
        zero_division=0,
    )[0]

    recall = recall_score(
        y_validation,
        predicted,
        labels=[TARGET_CLASS],
        average=None,
        zero_division=0,
    )[0]

    f1 = f1_score(
        y_validation,
        predicted,
        labels=[TARGET_CLASS],
        average=None,
        zero_division=0,
    )[0]

    validation_results.append(
        {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "file_transfer_f1": float(f1),
        }
    )

validation_results_df = pd.DataFrame(validation_results)

# Prefer the highest validation F1. In a tie, prefer the higher threshold.
best_row = validation_results_df.sort_values(
    ["file_transfer_f1", "threshold"],
    ascending=[False, False],
).iloc[0]

best_threshold = float(best_row["threshold"])

print("\nBest threshold selected using validation data:")
print(f"Threshold:       {best_threshold:.6f}")
print(f"Validation F1:   {best_row['file_transfer_f1']:.4f}")
print(f"Validation precision: {best_row['precision']:.4f}")
print(f"Validation recall:    {best_row['recall']:.4f}")

# Save validation threshold results separately.
validation_results_df.to_csv(OUTPUT_THRESHOLDS, index=False)

# --------------------------------------------------
# 7. Refit preprocessing and autoencoder on all
#    pre-cutoff training data
# --------------------------------------------------

print("\n" + "=" * 65)
print("PART B: FINAL TEMPORAL TEST EVALUATION")
print("=" * 65)

final_features, final_imputer, final_scaler, X_train = (
    fit_preprocessor(train_df, feature_columns)
)

X_test = transform_features(
    test_df, final_features, final_imputer, final_scaler
)

print(f"\nFinal retained features: {len(final_features)}")
print("Training final autoencoder on all training-period data...")

final_autoencoder, Z_train = train_autoencoder(
    X_train, epochs=EPOCHS
)

Z_test = encode_data(final_autoencoder, X_test)

final_classifier = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=SEED,
)

final_classifier.fit(Z_train, train_df[TARGET].to_numpy())

test_probabilities = final_classifier.predict_proba(Z_test)
final_class_order = list(final_classifier.classes_)

if TARGET_CLASS not in final_class_order:
    raise ValueError("file-transfer is missing from final classifier.")

final_target_index = final_class_order.index(TARGET_CLASS)
test_target_probabilities = test_probabilities[:, final_target_index]

# First create normal argmax predictions.
normal_predictions = np.array(final_class_order)[
    np.argmax(test_probabilities, axis=1)
]

# Then apply the threshold chosen from validation data.
threshold_predictions = normal_predictions.astype(object)
threshold_predictions[
    test_target_probabilities >= best_threshold
] = TARGET_CLASS

y_test = test_df[TARGET].to_numpy()

# --------------------------------------------------
# 8. Report metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, threshold_predictions)
macro_f1 = f1_score(
    y_test, threshold_predictions,
    labels=common_classes, average="macro", zero_division=0
)

ft_precision = precision_score(
    y_test, threshold_predictions,
    labels=[TARGET_CLASS], average=None, zero_division=0
)[0]

ft_recall = recall_score(
    y_test, threshold_predictions,
    labels=[TARGET_CLASS], average=None, zero_division=0
)[0]

ft_f1 = f1_score(
    y_test, threshold_predictions,
    labels=[TARGET_CLASS], average=None, zero_division=0
)[0]

report = classification_report(
    y_test,
    threshold_predictions,
    labels=common_classes,
    zero_division=0,
    digits=4,
)

summary = f"""
FILE-TRANSFER THRESHOLD-TUNING RESULTS
======================================

Temporal training rows: {len(train_df)}
Final test rows: {len(test_df)}
Validation rows: {len(validation_df)}
Validation start timestamp: {validation_start}

Classes: {common_classes}
Retained final features: {len(final_features)}
Autoencoder epochs: {EPOCHS}

Threshold selected using validation data: {best_threshold:.6f}

FINAL TEMPORAL TEST METRICS
---------------------------
Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)
Macro-F1: {macro_f1:.4f} ({macro_f1 * 100:.2f}%)

File-transfer precision: {ft_precision:.4f}
File-transfer recall:    {ft_recall:.4f}
File-transfer F1:        {ft_f1:.4f}

Classification report:
{report}

Actual file-transfer samples predicted as:
{pd.Series(
    threshold_predictions[y_test == TARGET_CLASS]
).value_counts().to_string()}

Predictions made as file-transfer, grouped by actual class:
{pd.Series(
    y_test[threshold_predictions == TARGET_CLASS]
).value_counts().to_string()}
"""

print(summary)

with open(OUTPUT_REPORT, "w", encoding="utf-8") as file:
    file.write(summary)

prediction_df = pd.DataFrame(
    {
        "actual": y_test,
        "predicted": threshold_predictions,
        "file_transfer_probability": test_target_probabilities,
    }
)

prediction_df.to_csv(OUTPUT_PREDICTIONS, index=False)

torch.save(
    final_autoencoder.state_dict(),
    OUTPUT_WEIGHTS,
)

print("\nSaved new experiment outputs:")
print(OUTPUT_REPORT)
print(OUTPUT_PREDICTIONS)
print(OUTPUT_THRESHOLDS)
print(OUTPUT_WEIGHTS)

print("\nExperiment complete.")
