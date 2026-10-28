
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

results_dir = Path("results")
predictions_path = results_dir / "fair_temporal_baseline_predictions.csv"
report_path = results_dir / "fair_temporal_baseline_updated.txt"

classes = [
    "audio-streaming",
    "browsing",
    "file-transfer",
    "p2p",
]

df = pd.read_csv(predictions_path)

if df[["actual", "predicted"]].isna().any().any():
    raise ValueError("Prediction file contains missing values.")

y_true = df["actual"]
y_pred = df["predicted"]

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=classes,
    average="macro",
    zero_division=0,
)

class_report = classification_report(
    y_true,
    y_pred,
    labels=classes,
    zero_division=0,
    digits=4,
)

report = f"""FAIR TEMPORAL BASELINE — VERIFIED PREDICTION REPORT

Training period: before 2016-02-24
Testing period: 2016-02-24 onward
Test rows: {len(df)}
Input features: 64

Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)
Macro-F1: {macro_f1:.4f} ({macro_f1 * 100:.2f}%)

Classification report:
{class_report}
"""

report_path.write_text(report, encoding="utf-8")

print(report)
print("Saved:", report_path)
print("Original fair_temporal_baseline.txt was not overwritten.")
