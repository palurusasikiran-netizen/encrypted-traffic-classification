
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
from sklearn.metrics import accuracy_score, f1_score, classification_report


# --------------------------------------------------
# 1. Reproducibility and configuration
# --------------------------------------------------
SEED = 42
EPOCHS = int(os.environ.get("AE_EPOCHS", "10"))

if not 5 <= EPOCHS <= 30:
    raise ValueError("AE_EPOCHS must be between 5 and 30.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
DATA_PATH = os.path.join("data", "Darknet.CSV")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Every epoch setting receives separate output files.
REPORT_PATH = os.path.join(
    RESULTS_DIR, f"self_supervised_{EPOCHS}epochs_report.txt"
)
PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR, f"self_supervised_{EPOCHS}epochs_predictions.csv"
)
WEIGHTS_PATH = os.path.join(
    RESULTS_DIR, f"autoencoder_{EPOCHS}epochs_weights.pt"
)

print("=" * 60)
print("SELF-SUPERVISED ENCRYPTED TRAFFIC CLASSIFICATION")
print("=" * 60)
print("Device:", DEVICE)
print("Random seed:", SEED)
print("Autoencoder epochs:", EPOCHS)
print("Loading dataset...")


# --------------------------------------------------
# 2. Load and prepare data
# --------------------------------------------------
df = pd.read_csv(DATA_PATH, low_memory=False)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d/%m/%Y %I:%M:%S %p",
    errors="coerce"
)

df["Label.1"] = (
    df["Label.1"].astype(str).str.strip().str.lower()
)

df = df.dropna(subset=["Timestamp", "Label.1"]).copy()

# Preserve classes represented in both time periods.
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
    "Flow ID", "Src IP", "Dst IP", "Timestamp",
    "Label", "Label.1"
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

# Replace infinite values with missing values.
X_train_raw = X_train_raw.replace([np.inf, -np.inf], np.nan)
X_test_raw = X_test_raw.replace([np.inf, -np.inf], np.nan)

# Remove columns entirely missing in the training data.
valid_columns = X_train_raw.columns[
    X_train_raw.notna().any(axis=0)
].tolist()

X_train_raw = X_train_raw[valid_columns]
X_test_raw = X_test_raw[valid_columns]

# Fit preprocessing using training data only.
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train_raw)
X_test = imputer.transform(X_test_raw)

# Remove features constant in the training data.
nonconstant = np.std(X_train, axis=0) > 0
X_train = X_train[:, nonconstant]
X_test = X_test[:, nonconstant]

# Scale features for neural-network training.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Clip extreme standardized values.
X_train = np.clip(X_train, -10, 10).astype(np.float32)
X_test = np.clip(X_test, -10, 10).astype(np.float32)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Features:", X_train.shape[1])
print("Common classes:", common_classes)


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
            nn.ReLU()
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
# 4. Train autoencoder without labels
# --------------------------------------------------
X_tensor = torch.from_numpy(X_train)

loader_generator = torch.Generator()
loader_generator.manual_seed(SEED)

loader = DataLoader(
    TensorDataset(X_tensor),
    batch_size=512,
    shuffle=True,
    generator=loader_generator
)

model = Autoencoder(X_train.shape[1]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

print(f"\nTraining autoencoder for {EPOCHS} epochs...")

epoch_losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_rows = 0

    for (batch,) in loader:
        batch = batch.to(DEVICE)

        optimizer.zero_grad()
        reconstructed = model(batch)
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
model.eval()

with torch.no_grad():
    train_embeddings = model.encoder(
        torch.from_numpy(X_train)
    ).numpy()

    test_embeddings = model.encoder(
        torch.from_numpy(X_test)
    ).numpy()

print("\nEmbedding dimensions:", train_embeddings.shape[1])


# --------------------------------------------------
# 6. Train classifier on learned representations
# --------------------------------------------------
print("\nTraining classifier on learned representations...")

classifier = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=SEED
)

classifier.fit(train_embeddings, y_train)
predictions = classifier.predict(test_embeddings)


# --------------------------------------------------
# 7. Evaluate the experiment
# --------------------------------------------------
accuracy = accuracy_score(y_test, predictions)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

report = classification_report(
    y_test,
    predictions,
    labels=common_classes,
    zero_division=0,
    digits=4
)

print("\n" + "=" * 60)
print("EXPERIMENT RESULTS")
print("=" * 60)
print("Epochs:", EPOCHS)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro-F1: {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")
print("\nClassification report:")
print(report)


# --------------------------------------------------
# 8. Save experiment-specific outputs
# --------------------------------------------------
with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write("SELF-SUPERVISED TEMPORAL-SHIFT EXPERIMENT\n")
    file.write("Dataset: CIC-Darknet2020 (Darknet.CSV)\n")
    file.write("Training period: before 2016-02-24\n")
    file.write("Testing period: 2016-02-24 onward\n")
    file.write(f"Random seed: {SEED}\n")
    file.write(f"Training rows: {len(X_train)}\n")
    file.write(f"Testing rows: {len(X_test)}\n")
    file.write(f"Input features: {X_train.shape[1]}\n")
    file.write(f"Latent dimensions: {train_embeddings.shape[1]}\n")
    file.write(f"Epochs: {EPOCHS}\n")
    file.write(f"Final reconstruction loss: {epoch_losses[-1]:.6f}\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Macro-F1: {macro_f1:.4f}\n\n")
    file.write("Reconstruction loss by epoch:\n")

    for index, loss_value in enumerate(epoch_losses, start=1):
        file.write(f"Epoch {index}: {loss_value:.6f}\n")

    file.write("\nClassification report:\n")
    file.write(report)

pd.DataFrame({
    "actual": y_test,
    "predicted": predictions
}).to_csv(PREDICTIONS_PATH, index=False)

torch.save(model.state_dict(), WEIGHTS_PATH)

print("\nSaved experiment files:")
print(REPORT_PATH)
print(PREDICTIONS_PATH)
print(WEIGHTS_PATH)
print("\nOriginal Day 1 result files were not overwritten.")


# --------------------------------------------------
# 9. Final summary for the dashboard
# --------------------------------------------------
print("\nEXPERIMENT COMPLETED")
print(f"Selected epochs: {EPOCHS}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro-F1: {macro_f1 * 100:.2f}%")
