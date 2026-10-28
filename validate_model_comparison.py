
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

folder = "results/"

baseline = pd.read_csv(
    folder + "fair_temporal_baseline_predictions.csv"
)
proposed = pd.read_csv(
    folder + "self_supervised_predictions.csv"
)
repeat = pd.read_csv(
    folder + "self_supervised_repeat_seed123_predictions.csv"
)

# Check that all models use the same test examples in the same order
assert len(baseline) == len(proposed) == len(repeat), (
    "Prediction files have different row counts."
)

assert baseline["actual"].equals(proposed["actual"]), (
    "Baseline and proposed actual labels do not match."
)

assert baseline["actual"].equals(repeat["actual"]), (
    "The two self-supervised runs use different test labels."
)

y_true = baseline["actual"]
classes = [
    "audio-streaming",
    "browsing",
    "file-transfer",
    "p2p"
]

models = {
    "Fair Temporal Baseline": baseline["predicted"],
    "Self-Supervised Seed 42": proposed["predicted"],
    "Self-Supervised Seed 123": repeat["predicted"]
}

print("All models evaluated on the same test examples.")
print("Test rows:", len(y_true))

summary = []

for name, predictions in models.items():
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    accuracy = accuracy_score(y_true, predictions)
    macro_f1 = f1_score(
        y_true, predictions, labels=classes,
        average="macro", zero_division=0
    )

    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro-F1: {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")

    print("\nClass-wise report:")
    print(classification_report(
        y_true, predictions,
        labels=classes,
        zero_division=0,
        digits=4
    ))

    summary.append({
        "Model": name,
        "Accuracy (%)": round(accuracy * 100, 4),
        "Macro-F1 (%)": round(macro_f1 * 100, 4)
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(
    folder + "validated_model_comparison.csv",
    index=False
)

print("\nVALIDATED SUMMARY")
print(summary_df.to_string(index=False))
print("\nSaved results/validated_model_comparison.csv")
