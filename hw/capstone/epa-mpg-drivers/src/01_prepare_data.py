"""
Step 1 of 2 - the same clean table as Report 2, plus make and model.

The features are exactly the eight from Report 2 (Student A's project). The only
change: we also keep make and model, so we can name a car when we explain
one. They are labels, not features - the model never sees them.

Run from the project folder:  python src/01_prepare_data.py
"""

import os
import urllib.request

import pandas as pd

URL = "https://www.fueleconomy.gov/feg/epadata/vehicles.csv"
RAW = os.path.join("data", "vehicles.csv")
CLEAN = os.path.join("data", "clean.csv")


def kind_of_vehicle(vclass):
    """Sort EPA's 30-odd size classes into four buckets (same as Report 1 and 2)."""
    text = str(vclass)
    if "Pickup" in text:
        return "Pickup"
    if "Sport Utility" in text:
        return "SUV"
    if "Cars" in text or "Two Seaters" in text or "Station Wagons" in text:
        return "Car"
    return "Van/other"


os.makedirs("data", exist_ok=True)
if not os.path.exists(RAW):
    print("Downloading the EPA file (about 21 MB)...")
    urllib.request.urlretrieve(URL, RAW)

df = pd.read_csv(RAW, low_memory=False)

# Gasoline only (electric cars report MPGe, a different unit), 1984-2025.
df = df[df["fuelType1"].astype(str).str.contains("Gasoline", na=False)]
df = df[df["year"] <= 2025]

# The eight spec-sheet features. city08 and highway08 stay out: comb08 is
# calculated from them, so they would leak the answer.
df["kind"] = df["VClass"].apply(kind_of_vehicle)
df["all_wheel"] = df["drive"].astype(str).str.contains("4-Wheel|All-Wheel").astype(int)
df["manual"] = df["trany"].astype(str).str.startswith("Manual").astype(int)
df["turbo"] = (df["tCharger"] == "T").astype(int)
df["hybrid"] = (df["atvType"] == "Hybrid").astype(int)

features = ["year", "displ", "cylinders", "all_wheel", "manual", "turbo",
            "hybrid", "kind"]
df = df[["make", "model"] + features + ["comb08"]]
df = df.dropna(subset=features)

df.to_csv(CLEAN, index=False)
print("Saved", CLEAN, "with", len(df), "rows")
