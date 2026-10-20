import pandas as pd

df = pd.read_csv("results/master_features_dataset.csv")

print(df["target_bpm"].describe())

print("\nBPM Range Counts")

low = (df["target_bpm"] < 75).sum()

medium = (
    (df["target_bpm"] >= 75) &
    (df["target_bpm"] <= 95)
).sum()

high = (df["target_bpm"] > 95).sum()

print(f"Low: {low}")
print(f"Medium: {medium}")
print(f"High: {high}")