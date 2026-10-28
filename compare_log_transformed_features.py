
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SEED = 42
EPOCHS = 10
BATCH_SIZE = 512
LATENT_DIM = 16

DATA_PATH = os.path.join("data", "Darknet.CSV")
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

REPORT_PATH = os.path.join(
    RESULTS_DIR, "log_transformed_features_report.txt"
)
PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR, "log_transformed_features_predictions.csv"
)
COMPARISON_PATH = os.path.join(
    RESULTS_DIR, "log_transformed_features_comparison.csv"
)
WEIGHTS_PATH = os.path.join(
    RESULTS_DIR, "log_transformed_autoencoder_weights.pt"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 65)
print("LOG-TRANSFORMED FEATURES VS ORIGINAL EMBEDDINGS")
print("=" * 65)

# --------------------------------------------------
# 2. Load the dataset and create the temporal split
# --------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH, low_memory=False)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d/%m/%Y %I:%M:%S %p",
    errors="coerce",
)

df["Label.1"] = (
    df["Label.1"].astype(str).str.strip().str.lower()
)

df = df.dropna(subset=["Timestamp", "Label.1"]).copy()

cutoff = pd.Timestamp("2016-02-24")

initial_train_mask = df["Timestamp"] < cutoff
initial_test_mask = df["Timestamp"] >= cutoff

common_classes = sorted(
    set(df.loc[initial_train_mask, "Label.1"])
    & set(df.loc[initial_test_mask, "Label.1"])
)

df = df[df["Label.1"].isin(common_classes)].copy()

train_mask = df["Timestamp"] < cutoff
test_mask = df["Timestamp"] >= cutoff

train_df = df.loc[train_mask].copy()
test_df = df.loc[test_mask].copy()

print(f"Training rows: {len(train_df):,}")
print(f"Test rows:     {len(test_df):,}")
print(f"Classes: {common_classes}")

excluded_columns = {
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
    "Label",
    "Label.1",
}

candidate_columns = [
    col for col in df.columns
    if col not in excluded_columns
    and pd.api.types.is_numeric_dtype(df[col])
]

X_train_raw = train_df[candidate_columns].copy()
X_test_raw = test_df[candidate_columns].copy()

y_train = train_df["Label.1"].to_numpy()
y_test = test_df["Label.1"].to_numpy()

# --------------------------------------------------
# 3. Numeric conversion and preprocessing
# --------------------------------------------------

X_train_raw = X_train_raw.apply(
    pd.to_numeric, errors="coerce"
).replace([np.inf, -np.inf], np.nan)

X_test_raw = X_test_raw.apply(
    pd.to_numeric, errors="coerce"
).replace([np.inf, -np.inf], np.nan)

# Remove features entirely missing in training data.
usable_columns = [
    col for col in X_train_raw.columns
    if X_train_raw[col].notna().any()
]

X_train_raw = X_train_raw[usable_columns]
X_test_raw = X_test_raw[usable_columns]

# Fit imputation on training data only.
imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train_raw)
X_test_imputed = imputer.transform(X_test_raw)

# Remove constant features based on training data only.
nonconstant = np.var(X_train_imputed, axis=0) > 0

X_train_imputed = X_train_imputed[:, nonconstant]
X_test_imputed = X_test_imputed[:, nonconstant]

retained_columns = [
    col for col, keep in zip(usable_columns, nonconstant)
    if keep
]

if X_train_imputed.shape[1] == 0:
    raise ValueError("No nonconstant numeric features remain.")

print(f"Numeric features retained: {len(retained_columns)}")

# --------------------------------------------------
# 4. Apply signed log transformation
# --------------------------------------------------

# sign(x) * log(1 + abs(x)) supports both positive
# and negative values while compressing large magnitudes.
X_train_log = (
    np.sign(X_train_imputed)
    * np.log1p(np.abs(X_train_imputed))
)

X_test_log = (
    np.sign(X_test_imputed)
    * np.log1p(np.abs(X_test_imputed))
)

# Fit the scaler on transformed training features only.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_log)
X_test_scaled = scaler.transform(X_test_log)

# Keep extreme standardized values bounded, as in the
# existing experiment pipeline.
X_train_scaled = np.clip(
    X_train_scaled, -10, 10
).astype(np.float32)

X_test_scaled = np.clip(
    X_test_scaled, -10, 10
).astype(np.float32)

# --------------------------------------------------
# 5. Define the autoencoder
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


# --------------------------------------------------
# 6. Train the autoencoder without labels
# --------------------------------------------------

print("\nTraining autoencoder on log-transformed features...")

torch.manual_seed(SEED)

X_tensor = torch.tensor(
    X_train_scaled, dtype=torch.float32
)

model = Autoencoder(
    input_dim=X_train_scaled.shape[1],
    latent_dim=LATENT_DIM,
)

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)

loss_fn = nn.MSELoss()

generator = torch.Generator()
generator.manual_seed(SEED)

loader = DataLoader(
    TensorDataset(X_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
)

for epoch in range(EPOCHS):
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

    print(
        f"Epoch {epoch + 1}/{EPOCHS}: "
        f"loss = {total_loss / total_rows:.6f}"
    )

# --------------------------------------------------
# 7. Generate embeddings
# --------------------------------------------------

print("\nGenerating embeddings...")

model.eval()

with torch.no_grad():
    Z_train = model.encoder(
        torch.tensor(X_train_scaled, dtype=torch.float32)
    ).numpy()

    Z_test = model.encoder(
        torch.tensor(X_test_scaled, dtype=torch.float32)
    ).numpy()

# --------------------------------------------------
# 8. Train the supervised downstream classifier
# --------------------------------------------------

print("Training HistGradientBoostingClassifier...")

classifier = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=SEED,
)

classifier.fit(Z_train, y_train)

predictions = classifier.predict(Z_test)

# --------------------------------------------------
# 9. Evaluate the untouched temporal test set
# --------------------------------------------------

accuracy = accuracy_score(y_test, predictions)

macro_f1 = f1_score(
    y_test,
    predictions,
    labels=common_classes,
    average="macro",
    zero_division=0,
)

report = classification_report(
    y_test,
    predictions,
    labels=common_classes,
    zero_division=0,
    digits=4,
)

ft_precision = f1_score(
    y_test,
    predictions,
    labels=["file-transfer"],
    average=None,
    zero_division=0,
)

from sklearn.metrics import precision_score, recall_score

ft_precision_value = precision_score(
    y_test,
    predictions,
    labels=["file-transfer"],
    average=None,
    zero_division=0,
)[0]

ft_recall_value = recall_score(
    y_test,
    predictions,
    labels=["file-transfer"],
    average=None,
    zero_division=0,
)[0]

ft_f1_value = ft_precision[0]

summary = f"""
LOG-TRANSFORMED FEATURE EXPERIMENT
==================================

Training rows: {len(train_df)}
Test rows: {len(test_df)}
Features retained: {len(retained_columns)}
Autoencoder epochs: {EPOCHS}
Latent dimension: {LATENT_DIM}

Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)
Macro-F1: {macro_f1:.4f} ({macro_f1 * 100:.2f}%)

File-transfer precision: {ft_precision_value:.4f}
File-transfer recall:    {ft_recall_value:.4f}
File-transfer F1:        {ft_f1_value:.4f}

Classification report:
{report}
"""

print("\n" + "=" * 65)
print("FINAL TEST RESULTS")
print("=" * 65)
print(summary)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(summary)

pd.DataFrame({
    "actual": y_test,
    "predicted": predictions,
}).to_csv(PREDICTIONS_PATH, index=False)

pd.DataFrame([{
    "model": "log_transformed_autoencoder_embeddings",
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "file_transfer_precision": ft_precision_value,
    "file_transfer_recall": ft_recall_value,
    "file_transfer_f1": ft_f1_value,
}]).to_csv(COMPARISON_PATH, index=False)

torch.save(model.state_dict(), WEIGHTS_PATH)

print("Saved new experiment files:")
print(REPORT_PATH)
print(PREDICTIONS_PATH)
print(COMPARISON_PATH)
print(WEIGHTS_PATH)

print("\nExperiment completed.")
