
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
DATA_PATH = os.path.join("data", "Darknet.CSV")
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

REPORT_PATH = os.path.join(
    RESULTS_DIR, "raw_vs_embeddings_report.txt"
)
PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR, "raw_vs_embeddings_predictions.csv"
)
COMPARISON_PATH = os.path.join(
    RESULTS_DIR, "raw_vs_embeddings_comparison.csv"
)
WEIGHTS_PATH = os.path.join(
    RESULTS_DIR, "raw_vs_embeddings_autoencoder_weights.pt"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 65)
print("RAW FEATURES VS AUTOENCODER EMBEDDINGS")
print("=" * 65)


# --------------------------------------------------
# 2. Load and prepare the dataset
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

X_train_raw = df.loc[
    train_mask, candidate_columns
].copy()

X_test_raw = df.loc[
    test_mask, candidate_columns
].copy()

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

# Fit the imputer on training data only.
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train_raw)
X_test = imputer.transform(X_test_raw)

# Remove features that are constant in training data.
nonconstant = np.std(X_train, axis=0) > 0

X_train = X_train[:, nonconstant]
X_test = X_test[:, nonconstant]

# Fit the scaler on training data only.
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = np.clip(X_train, -10, 10).astype(np.float32)
X_test = np.clip(X_test, -10, 10).astype(np.float32)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Raw feature dimensions:", X_train.shape[1])
print("Classes:", common_classes)


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
# 4. Train the autoencoder without labels
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
# 5. Extract 16-dimensional embeddings
# --------------------------------------------------
autoencoder.eval()

with torch.no_grad():
    train_embeddings = autoencoder.encoder(
        torch.from_numpy(X_train)
    ).numpy()

    test_embeddings = autoencoder.encoder(
        torch.from_numpy(X_test)
    ).numpy()

torch.save(
    autoencoder.state_dict(), WEIGHTS_PATH
)

print("\nRaw feature dimensions:", X_train.shape[1])
print("Embedding dimensions:", train_embeddings.shape[1])


# --------------------------------------------------
# 6. Train both classifiers
# --------------------------------------------------
# Both classifiers use the same algorithm and settings.
# Neither classifier uses class weights in this comparison.

print("\nTraining classifier on raw features...")

raw_classifier = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=SEED,
)

raw_classifier.fit(X_train, y_train)
raw_predictions = raw_classifier.predict(X_test)


print("\nTraining classifier on autoencoder embeddings...")

embedding_classifier = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=SEED,
)

embedding_classifier.fit(train_embeddings, y_train)
embedding_predictions = embedding_classifier.predict(
    test_embeddings
)


# --------------------------------------------------
# 7. Evaluate both models
# --------------------------------------------------
def evaluate_model(name, actual, predicted):
    accuracy = accuracy_score(actual, predicted)

    macro_f1 = f1_score(
        actual,
        predicted,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        actual,
        predicted,
        labels=common_classes,
        zero_division=0,
        digits=4,
    )

    print("\n" + "-" * 65)
    print(name)
    print("-" * 65)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro-F1: {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")
    print("\nClassification report:")
    print(report)

    return {
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "report": report,
    }


raw_result = evaluate_model(
    "RAW 64-FEATURE CLASSIFIER",
    y_test,
    raw_predictions,
)

embedding_result = evaluate_model(
    "AUTOENCODER 16-DIMENSIONAL EMBEDDING CLASSIFIER",
    y_test,
    embedding_predictions,
)


# --------------------------------------------------
# 8. Create a comparison table
# --------------------------------------------------
comparison_rows = []

for result in [raw_result, embedding_result]:
    report_dict = classification_report(
        y_test,
        raw_predictions if result["model"].startswith("RAW")
        else embedding_predictions,
        labels=common_classes,
        output_dict=True,
        zero_division=0,
    )

    row = {
        "model": result["model"],
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
    }

    for class_name in common_classes:
        row[f"{class_name}_precision"] = (
            report_dict[class_name]["precision"]
        )
        row[f"{class_name}_recall"] = (
            report_dict[class_name]["recall"]
        )
        row[f"{class_name}_f1"] = (
            report_dict[class_name]["f1-score"]
        )

    comparison_rows.append(row)

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(COMPARISON_PATH, index=False)

predictions_df = pd.DataFrame({
    "actual": y_test,
    "raw_predicted": raw_predictions,
    "embedding_predicted": embedding_predictions,
})

predictions_df.to_csv(PREDICTIONS_PATH, index=False)


# --------------------------------------------------
# 9. Save a complete text report
# --------------------------------------------------
with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write("RAW FEATURES VS AUTOENCODER EMBEDDINGS\n")
    file.write("Dataset: CIC-Darknet2020 (Darknet.CSV)\n")
    file.write("Training period: before 2016-02-24\n")
    file.write("Testing period: 2016-02-24 onward\n")
    file.write(f"Random seed: {SEED}\n")
    file.write(f"Autoencoder epochs: {EPOCHS}\n")
    file.write(f"Training rows: {len(X_train)}\n")
    file.write(f"Testing rows: {len(X_test)}\n")
    file.write(f"Raw dimensions: {X_train.shape[1]}\n")
    file.write(f"Embedding dimensions: {train_embeddings.shape[1]}\n")
    file.write(
        f"Final reconstruction loss: {epoch_losses[-1]:.6f}\n\n"
    )

    for result in [raw_result, embedding_result]:
        file.write("=" * 65 + "\n")
        file.write(result["model"] + "\n")
        file.write("=" * 65 + "\n")
        file.write(f"Accuracy: {result['accuracy']:.6f}\n")
        file.write(f"Macro-F1: {result['macro_f1']:.6f}\n\n")
        file.write(result["report"] + "\n")

    file.write("\nCOMPARISON TABLE\n")
    file.write(comparison_df.to_string(index=False))
    file.write("\n")

print("\n" + "=" * 65)
print("FINAL COMPARISON")
print("=" * 65)
print(comparison_df.to_string(index=False))

print("\nSaved experiment files:")
print(REPORT_PATH)
print(PREDICTIONS_PATH)
print(COMPARISON_PATH)
print(WEIGHTS_PATH)

print("\nEXPERIMENT COMPLETED")
print("Existing experiment files were not overwritten.")
