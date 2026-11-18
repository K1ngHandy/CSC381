#  step 1 - clean the Kaggle student dropout file

import os
import pandas as pd

RAW = os.path.join("data", "student_dropout.csv")
CLEAN = os.path.join("data", "clean.csv")

df = pd.read_csv(RAW)
print("Shape:", df.shape)
print("Dropout rate:", round(df["dropout"].mean(), 3))


df = df.drop(columns=["person_id"])

#  basic checks
print("\nMissing values per column (top 10):")
print(df.isna().sum().sort_values(ascending=False).head(10))

#  keep all features
df.to_csv(CLEAN, index=False)
print("Saved", CLEAN, "with", len(df), "rows")