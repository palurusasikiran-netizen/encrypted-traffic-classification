
import pandas as pd
from pathlib import Path

path = Path("data") / "Darknet.CSV"

# Read only the columns needed for this analysis.
df = pd.read_csv(
    path,
    usecols=["Timestamp", "Label.1"],
    low_memory=False
)

df["Label.1"] = (
    df["Label.1"].astype("string").str.strip().str.lower()
)

# Match the timestamp format shown in the dataset.
df["time"] = pd.to_datetime(
    df["Timestamp"],
    format="%d/%m/%Y %I:%M:%S %p",
    errors="coerce"
)

print("Rows:", len(df))
print("Invalid timestamps:", df["time"].isna().sum())
print("Earliest timestamp:", df["time"].min())
print("Latest timestamp:", df["time"].max())

df = df.dropna(subset=["time", "Label.1"])
df["date"] = df["time"].dt.date

print("\nNumber of records per date:")
print(df.groupby("date").size().to_string())

print("\nApplication classes represented per date:")
daily_classes = df.groupby("date")["Label.1"].nunique()
print(daily_classes.to_string())

print("\nDaily application distribution (counts):")
daily_counts = pd.crosstab(df["date"], df["Label.1"])
print(daily_counts.to_string())

print("\nAnalysis complete.")
