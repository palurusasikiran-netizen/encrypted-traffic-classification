
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Load the verified model comparison data
source_path = RESULTS_DIR / "validated_model_comparison.csv"
results = pd.read_csv(source_path)

# Display the exact values used
print("\nFINAL MODEL COMPARISON\n")
print(results.to_string(index=False))

# Save a copy of the final comparison
csv_path = RESULTS_DIR / "final_model_comparison_verified.csv"
results.to_csv(csv_path, index=False)

# Generate the final comparison chart
ax = results.plot(
    x="Model",
    y=["Accuracy (%)", "Macro-F1 (%)"],
    kind="bar",
    figsize=(11, 6),
    rot=12
)

ax.set_title("Model Performance Under Temporal Distribution Shift")
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3)

plt.tight_layout()

chart_path = RESULTS_DIR / "final_model_comparison_verified.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved:")
print(csv_path)
print(chart_path)
