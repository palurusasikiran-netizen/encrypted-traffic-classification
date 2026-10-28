
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Read the verified comparison data
data = pd.read_csv(
    os.path.join(RESULTS_DIR, "validated_model_comparison.csv")
)

models = data["Model"].tolist()
accuracy = data["Accuracy (%)"].to_numpy()
macro_f1 = data["Macro-F1 (%)"].to_numpy()

x = np.arange(len(models))
width = 0.36

fig, ax = plt.subplots(figsize=(11, 6))

accuracy_bars = ax.bar(
    x - width / 2, accuracy, width, label="Accuracy"
)
f1_bars = ax.bar(
    x + width / 2, macro_f1, width, label="Macro-F1"
)

ax.set_ylabel("Score (%)")
ax.set_title(
    "Encrypted Traffic Classification Under Temporal Shift"
)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=10, ha="right")
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis="y", alpha=0.3)

ax.bar_label(accuracy_bars, fmt="%.2f%%", padding=3)
ax.bar_label(f1_bars, fmt="%.2f%%", padding=3)

fig.tight_layout()

output_path = os.path.join(
    RESULTS_DIR, "validated_model_comparison.png"
)
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", output_path)
print(data.to_string(index=False))
