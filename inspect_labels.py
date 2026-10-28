
import pandas as pd
from pathlib import Path

file_path = Path("data") / "Darknet.CSV"

# Read only the label columns first
df = pd.read_csv(
    file_path,
    usecols=["Label", "Label.1"],
    low_memory=False
)

print("=" * 60)
print("LABEL ANALYSIS")
print("=" * 60)

for column in ["Label", "Label.1"]:
    print(f"\n--- {column} ---")

    # Clean whitespace and count missing values
    labels = df[column].astype("string").str.strip()
    print("Missing values:", labels.isna().sum())
    print("Unique classes:", labels.nunique(dropna=True))
    print("\nClass counts:")
    print(labels.value_counts(dropna=False).to_string())

print("\n" + "=" * 60)
print("LABEL COMBINATION ANALYSIS")
print("=" * 60)

print("\nNumber of distinct label combinations:")
print(df.drop_duplicates().shape[0])

print("\nLabel combination counts:")
print(
    df.value_counts(dropna=False)
      .head(30)
      .to_string()
)

print("\nAnalysis completed.")
