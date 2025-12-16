import pandas as pd

df = pd.read_csv("data/processed/processed_dataset.csv")
print(df.head(20))
print("\nColumns:\n", df.columns.tolist())
