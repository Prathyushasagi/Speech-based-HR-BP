import pandas as pd


bpm_path = r"E:\Speech for HR & BP\Voice_HR_Project\data\bpm\1.csv"

df = pd.read_csv(bpm_path)

print(df.head())

print("\nColumns:")
print(df.columns)

print("\nShape:")
print(df.shape)