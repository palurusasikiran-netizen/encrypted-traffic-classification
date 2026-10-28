
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

RESULTS_DIR = "results"

models = {
    "Temporal baseline": "fair_temporal_baseline_predictions.csv",
    "Self-supervised (Seed 42)": "self_supervised_predictions.csv",
}

classes = ["audio-streaming", "browsing", "file-transfer", "p2p"]
scores = {}

for model_name, filename in models.items():
    path = os.path.join(RESULTS_DIR, filename)
    predictions = pd.read_csv(path)

    scores[model_name] = [
        f1_score(
            predictions["actual"],
            predictions["predicted"],
            labels=[class_name],
            average="macro",
            zero_division=0,
        )
        for class_name in classes
    ]

x = np.arange(len(classes))
width = 0.36

fig, ax = plt.subplots(figsize=(10, 6))

baseline_bars = ax.bar(
    x - width / 2,
    scores["Temporal baseline"],
    width,
    label="Temporal baseline",
)
self_supervised_bars = ax.bar(
    x + width / 2,
    scores["Self-supervised (Seed 42)"],
    width,
    label="Self-supervised (Seed 42)",
)

ax.set_ylabel("F1-score")
ax.set_title("Per-Class F1 Under Temporal Distribution Shift")
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", alpha=0.3)

ax.bar_label(baseline_bars, fmt="%.3f", padding=3)
ax.bar_label(self_supervised_bars, fmt="%.3f", padding=3)

fig.tight_layout()

output_path = os.path.join(
    RESULTS_DIR, "validated_class_f1_comparison.png"
)
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", output_path)
print(pd.DataFrame({
    "Class": classes,
    "Temporal baseline F1": scores["Temporal baseline"],
    "Self-supervised Seed 42 F1": scores["Self-supervised (Seed 42)"],
}).to_string(index=False))
