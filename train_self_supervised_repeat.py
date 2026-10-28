
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


# -----------------------------
# 1. Reproducibility and setup
# -----------------------------
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
DATA_PATH = os.path.join("data", "Darknet.CSV")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Device:", DEVICE)
print("Loading dataset...")


# -----------------------------
# 2. Load and prepare data
# -----------------------------
df = pd.read_csv(DATA_PATH, low_memory=False)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d/%m/%Y %I:%M:%S %p",
    errors="coerce"
)

df["Label.1"] = df["Label.1"].astype(str).str.strip().str.lower()

df = df.dropna(subset=["Timestamp", "Label.1"]).copy()

# Keep only classes represented in BOTH periods.
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

# Convert infinities to missing values.
X_train_raw = X_train_raw.replace([np.inf, -np.inf], np.nan)
X_test_raw = X_test_raw.replace([np.inf, -np.inf], np.nan)

# Remove columns that are entirely missing in training.
valid_columns = X_train_raw.columns[
    X_train_raw.notna().any(axis=0)
].tolist()

X_train_raw = X_train_raw[valid_columns]
X_test_raw = X_test_raw[valid_columns]

# Fit preprocessing ONLY on training data.
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train_raw)
X_test = imputer.transform(X_test_raw)

# Remove features that are constant in training.
nonconstant = np.std(X_train, axis=0) > 0

X_train = X_train[:, nonconstant]
X_test = X_test[:, nonconstant]

# Scale for neural-network training.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Limit extreme standardized values for stable reconstruction training.
X_train = np.clip(X_train, -10, 10).astype(np.float32)
X_test = np.clip(X_test, -10, 10).astype(np.float32)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Features:", X_train.shape[1])
print("Common classes:", common_classes)


# -----------------------------
# 3. Self-supervised autoencoder
# -----------------------------
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


X_tensor = torch.from_numpy(X_train)

loader = DataLoader(
    TensorDataset(X_tensor),
    batch_size=512,
    shuffle=True
)

model = Autoencoder(X_train.shape[1]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

EPOCHS = 10

print("\nTraining autoencoder without labels...")

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
    print(f"Epoch {epoch + 1}/{EPOCHS} - "
          f"reconstruction loss: {average_loss:.6f}")


# -----------------------------
# 4. Extract learned features
# -----------------------------
model.eval()

with torch.no_grad():
    train_embeddings = model.encoder(
        torch.from_numpy(X_train)
    ).numpy()

    test_embeddings = model.encoder(
        torch.from_numpy(X_test)
    ).numpy()


# -----------------------------
# 5. Train classifier on embeddings
# -----------------------------
print("\nTraining classifier on learned representations...")

classifier = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=SEED
)

classifier.fit(train_embeddings, y_train)
predictions = classifier.predict(test_embeddings)


# -----------------------------
# 6. Evaluate and save results
# -----------------------------
accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test, predictions,
    average="macro",
    zero_division=0
)

report = classification_report(
    y_test,
    predictions,
    labels=common_classes,
    zero_division=0
)

output_path = os.path.join(
    RESULTS_DIR, "self_supervised_repeat_seed123_report.txt"
)

with open(output_path, "w", encoding="utf-8") as file:
    file.write("SELF-SUPERVISED TEMPORAL-SHIFT EXPERIMENT\n")
    file.write("Training period: before 2016-02-24\n")
    file.write("Testing period: 2016-02-24 onward\n")
    file.write(f"Training rows: {len(X_train)}\n")
    file.write(f"Testing rows: {len(X_test)}\n")
    file.write(f"Input features: {X_train.shape[1]}\n")
    file.write(f"Latent dimensions: {train_embeddings.shape[1]}\n")
    file.write(f"Epochs: {EPOCHS}\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Macro-F1: {macro_f1:.4f}\n\n")
    file.write(report)

pd.DataFrame({
    "actual": y_test,
    "predicted": predictions
}).to_csv(
    os.path.join(RESULTS_DIR, "self_supervised_repeat_seed123_predictions.csv"),
    index=False
)

torch.save(
    model.state_dict(),
    os.path.join(RESULTS_DIR, "autoencoder_repeat_seed123_weights.pt")
)

print("\nSELF-SUPERVISED TEMPORAL-SHIFT RESULTS")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")
print(report)
print("Saved:", output_path)
