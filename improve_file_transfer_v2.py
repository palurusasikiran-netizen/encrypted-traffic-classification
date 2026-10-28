
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
from sklearn.utils.class_weight import compute_sample_weight


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
SEED = 42
EPOCHS = int(os.environ.get("AE_EPOCHS", "10"))
FILE_TRANSFER_BOOST = 2.0

if not 5 <= EPOCHS <= 30:
    raise ValueError("AE_EPOCHS must be between 5 and 30.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = os.path.join("data", "Darknet.CSV")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

REPORT_PATH = os.path.join(
    RESULTS_DIR, f"file_transfer_v2_{EPOCHS}epochs_report.txt"
)
PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR, f"file_transfer_v2_{EPOCHS}epochs_predictions.csv"
)
WEIGHTS_PATH = os.path.join(
    RESULTS_DIR, f"file_transfer_v2_{EPOCHS}epochs_weights.pt"
)

print("=" * 65)
print("TARGETED FILE-TRANSFER CLASSIFICATION EXPERIMENT")
print("=" * 65)
print("Device: CPU")
print("Random seed:", SEED)
print("Autoencoder epochs:", EPOCHS)
print("File-transfer weight multiplier:", FILE_TRANSFER_BOOST)


# --------------------------------------------------
# 2. Load and prepare data
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
initial_test_mask = ~initial_train_mask

common_classes = sorted(
    set(df.loc[initial_train_mask, "Label.1"])
    & set(df.loc[initial_test_mask, "Label.1"])
)

df = df[df["Label.1"].isin(common_classes)].copy()

train_mask = df["Timestamp"] < cutoff
test_mask = ~train_mask

excluded = {
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
    "Label",
    "Label.1",
}

candidate_columns = [
    col for col in df.columns
    if col not in excluded
    and pd.api.types.is_numeric_dtype(df[col])
]

X_train_raw = df.loc[train_mask, candidate_columns].copy()
X_test_raw = df.loc[test_mask, candidate_columns].copy()

y_train = df.loc[train_mask, "Label.1"].to_numpy()
y_test = df.loc[test_mask, "Label.1"].to_numpy()

X_train_raw = X_train_raw.replace(
    [np.inf, -np.inf], np.nan
)
X_test_raw = X_test_raw.replace(
    [np.inf, -np.inf], np.nan
)

valid_columns = X_train_raw.columns[
    X_train_raw.notna().any(axis=0)
].tolist()

X_train_raw = X_train_raw[valid_columns]
X_test_raw = X_test_raw[valid_columns]

# Fit preprocessing only on training data.
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train_raw)
X_test = imputer.transform(X_test_raw)

# Remove features constant in training data.
nonconstant = np.std(X_train, axis=0) > 0

X_train = X_train[:, nonconstant]
X_test = X_test[:, nonconstant]

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = np.clip(X_train, -10, 10).astype(np.float32)
X_test = np.clip(X_test, -10, 10).astype(np.float32)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Input features:", X_train.shape[1])
print("Classes:", common_classes)

print("\nTraining class distribution:")
print(pd.Series(y_train).value_counts().to_string())


# --------------------------------------------------
# 3. Define the autoencoder
# --------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)


# --------------------------------------------------
# 4. Train autoencoder without labels
# --------------------------------------------------
print(f"\nTraining autoencoder for {EPOCHS} epochs...")

X_tensor = torch.from_numpy(X_train)

loader_generator = torch.Generator()
loader_generator.manual_seed(SEED)

loader = DataLoader(
    TensorDataset(X_tensor),
    batch_size=512,
    shuffle=True,
    generator=loader_generator,
)

autoencoder = Autoencoder(X_train.shape[1])

optimizer = torch.optim.Adam(
    autoencoder.parameters(), lr=0.001
)

loss_function = nn.MSELoss()
epoch_losses = []

for epoch in range(EPOCHS):
    autoencoder.train()

    total_loss = 0.0
    total_rows = 0

    for (batch,) in loader:
        optimizer.zero_grad()

        reconstructed = autoencoder(batch)
        loss = loss_function(reconstructed, batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch)
        total_rows += len(batch)

    average_loss = total_loss / total_rows
    epoch_losses.append(average_loss)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} - "
        f"reconstruction loss: {average_loss:.6f}"
    )


# --------------------------------------------------
# 5. Extract learned representations
# --------------------------------------------------
autoencoder.eval()

with torch.no_grad():
    train_embeddings = autoencoder.encoder(
        torch.from_numpy(X_train)
    ).numpy()

    test_embeddings = autoencoder.encoder(
        torch.from_numpy(X_test)
    ).numpy()

print("\nEmbedding dimensions:", train_embeddings.shape[1])


# --------------------------------------------------
# 6. Train classifier with extra file-transfer weight
# --------------------------------------------------
print("\nCalculating balanced sample weights...")

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

# Double the balanced weight assigned to file-transfer.
file_transfer_mask = y_train == "file-transfer"
sample_weights[file_transfer_mask] *= FILE_TRANSFER_BOOST

print("File-transfer weight multiplier:", FILE_TRANSFER_BOOST)
print("Mean file-transfer sample weight:",
      round(float(sample_weights[file_transfer_mask].mean()), 4))

print("\nTraining classifier...")

classifier = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=SEED,
)

classifier.fit(
    train_embeddings,
    y_train,
    sample_weight=sample_weights,
)

predictions = classifier.predict(test_embeddings)


# --------------------------------------------------
# 7. Evaluate the model
# --------------------------------------------------
accuracy = accuracy_score(y_test, predictions)

macro_f1 = f1_score(
    y_test,
    predictions,
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

diagnostic = pd.DataFrame({
    "actual": y_test,
    "predicted": predictions,
})

print("\n" + "=" * 65)
print("TARGETED EXPERIMENT RESULTS")
print("=" * 65)

print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro-F1: {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")

print("\nClassification report:")
print(report)

print("\nActual file-transfer samples predicted as:")
actual_ft = diagnostic[
    diagnostic["actual"] == "file-transfer"
]

print(actual_ft["predicted"].value_counts().to_string())

print("\nActual classes among file-transfer predictions:")
predicted_ft = diagnostic[
    diagnostic["predicted"] == "file-transfer"
]

if len(predicted_ft) > 0:
    print(predicted_ft["actual"].value_counts().to_string())
else:
    print("No file-transfer predictions were made.")


# --------------------------------------------------
# 8. Save separate experiment results
# --------------------------------------------------
with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write("TARGETED FILE-TRANSFER WEIGHTING EXPERIMENT\n")
    file.write("Dataset: CIC-Darknet2020 (Darknet.CSV)\n")
    file.write("Training period: before 2016-02-24\n")
    file.write("Testing period: 2016-02-24 onward\n")
    file.write(f"Random seed: {SEED}\n")
    file.write(f"Epochs: {EPOCHS}\n")
    file.write(f"File-transfer weight multiplier: {FILE_TRANSFER_BOOST}\n")
    file.write(f"Training rows: {len(X_train)}\n")
    file.write(f"Testing rows: {len(X_test)}\n")
    file.write(f"Input features: {X_train.shape[1]}\n")
    file.write(f"Latent dimensions: {train_embeddings.shape[1]}\n")
    file.write(f"Final reconstruction loss: {epoch_losses[-1]:.6f}\n")
    file.write(f"Accuracy: {accuracy:.6f}\n")
    file.write(f"Macro-F1: {macro_f1:.6f}\n\n")
    file.write("Classification report:\n")
    file.write(report)
    file.write("\n\nActual file-transfer samples predicted as:\n")
    file.write(actual_ft["predicted"].value_counts().to_string())
    file.write("\n\nActual classes among file-transfer predictions:\n")

    if len(predicted_ft) > 0:
        file.write(predicted_ft["actual"].value_counts().to_string())
    else:
        file.write("No file-transfer predictions were made.")

    file.write("\n\nReconstruction loss by epoch:\n")

    for index, loss_value in enumerate(epoch_losses, start=1):
        file.write(f"Epoch {index}: {loss_value:.6f}\n")

diagnostic.to_csv(PREDICTIONS_PATH, index=False)
torch.save(autoencoder.state_dict(), WEIGHTS_PATH)

print("\nSaved experiment files:")
print(REPORT_PATH)
print(PREDICTIONS_PATH)
print(WEIGHTS_PATH)

print("\nEXPERIMENT COMPLETED")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro-F1: {macro_f1 * 100:.2f}%")
print("Existing experiment files were not overwritten.")
