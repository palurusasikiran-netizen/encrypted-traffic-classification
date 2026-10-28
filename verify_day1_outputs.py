
from pathlib import Path
import pandas as pd

results = Path("results")

files_to_check = [
    "baseline_predictions.csv",
    "baseline_report.csv",
    "controlled_shift_report.txt",
    "fair_temporal_baseline.txt",
    "self_supervised_predictions.csv",
    "self_supervised_report.txt",
    "self_supervised_repeat_seed123_predictions.csv",
    "self_supervised_repeat_seed123_report.txt",
    "autoencoder_weights.pt",
    "autoencoder_repeat_seed123_weights.pt",
    "final_model_comparison.csv",
    "final_model_comparison.png",
    "class_f1_comparison.png",
]

print("=" * 55)
print("DAY 1 OUTPUT VERIFICATION")
print("=" * 55)

missing = []

for filename in files_to_check:
    path = results / filename

    if path.exists() and path.stat().st_size > 0:
        print(f"[OK] {filename} ({path.stat().st_size:,} bytes)")
    else:
        print(f"[MISSING OR EMPTY] {filename}")
        missing.append(filename)

print("\n" + "=" * 55)
print("PREDICTION FILE INSPECTION")
print("=" * 55)

prediction_files = [
    "baseline_predictions.csv",
    "self_supervised_predictions.csv",
    "self_supervised_repeat_seed123_predictions.csv",
]

for filename in prediction_files:
    path = results / filename

    if not path.exists():
        continue

    try:
        df = pd.read_csv(path)
        print(f"\n{filename}")
        print("Rows:", len(df))
        print("Columns:", list(df.columns))
        print("Missing values:", int(df.isna().sum().sum()))
        print("Sample:")
        print(df.head(3).to_string(index=False))
    except Exception as error:
        print(f"Could not inspect {filename}: {error}")

print("\n" + "=" * 55)
if missing:
    print("Some expected outputs need attention.")
else:
    print("All listed output files exist and are non-empty.")
print("Note: file checks do not prove model correctness.")
