"""Lecture 7: a driver analysis on real hotel bookings. What drives cancellations?
Put hotel_bookings.csv in the same folder, then run:
    python L7_driver_examples.py
You need the shap package (pip install shap). Charts open in a window: close each one to continue.
Data: 119,390 bookings at two hotels in Portugal, 2015-2017 (Antonio, Almeida, Nunes, 2019).
"""

# ---------------------------------------------------------------- [1] load
import pandas as pd

df = pd.read_csv("hotel_bookings.csv")     # one row = one booking
print(df.shape)                            # (bookings, columns)
print("Canceled:", round(df["is_canceled"].mean(), 3))   # 1 = canceled

# Keep the 10 biggest countries; call the rest "Other"
big = df["country"].value_counts().index[:10]
df["country"] = df["country"].where(df["country"].isin(big), "Other")

# ---------------------------------------------------------------- [2] features
not_features = ["is_canceled",                # the answer itself
                "reservation_status",         # says "Canceled": the answer again
                "reservation_status_date",
                "assigned_room_type",         # only known at check-in
                "booking_changes",            # happens after the booking
                "agent", "company",           # ID numbers, not facts
                "arrival_date_year", "arrival_date_week_number",
                "arrival_date_day_of_month"]
X = df.drop(columns=not_features)
X = pd.get_dummies(X, dtype=float)         # words -> 0/1 columns
X = X.fillna(0)                            # fill missing values
y = df["is_canceled"]
print(X.shape[1], "features")

# ---------------------------------------------------------------- [3] forest
from sklearn.ensemble import RandomForestClassifier

# Learn from 2015-2016, check on 2017
train = df["arrival_date_year"] <= 2016
valid = df["arrival_date_year"] == 2017
X_train, y_train = X[train], y[train]
X_valid, y_valid = X[valid], y[valid]

forest = RandomForestClassifier(n_estimators=300,     # 300 trees
                                min_samples_leaf=50,  # no tiny leaves
                                max_features=0.33,    # each split sees 1/3
                                random_state=0,       # same result every run
                                n_jobs=-1)            # use all CPU cores
forest.fit(X_train, y_train)                          # learn from 2015-2016

# ---------------------------------------------------------------- [4] score
guess_no = 1 - y_valid.mean()              # always say "will not cancel"
start_score = forest.score(X_valid, y_valid)          # share we get right
print("Always say no:", round(guess_no, 3))
print("Forest:       ", round(start_score, 3))

# ---------------------------------------------------------------- [5] shuffle
import numpy as np

def shuffle_test(name):                    # name = the start of the column name
    columns = [c for c in X.columns if c.startswith(name)]
    drops = []
    for seed in range(5):                  # shuffle 5 times
        order = np.random.RandomState(seed).permutation(len(X_valid))
        shuffled = X_valid.copy()          # a copy: the real data stays safe
        shuffled[columns] = X_valid[columns].values[order]
        drops.append(start_score - forest.score(shuffled, y_valid))
    return round(np.mean(drops), 3)        # average drop in the score

facts = ["deposit_type", "country", "lead_time", "total_of_special_requests",
         "market_segment", "required_car_parking_spaces", "customer_type",
         "previous_cancellations", "adr", "hotel"]
for fact in facts:
    print(fact, shuffle_test(fact))

# ---------------------------------------------------------------- [6] shap_all
import shap

sample = X_valid.sample(1000, random_state=0)   # 1,000 bookings from 2017
explainer = shap.TreeExplainer(forest)          # a fast explainer for tree models
explanation = explainer(sample)                 # takes about a minute
explanation = explanation[:, :, 1]              # pushes toward "canceled"

print("Starting point:", round(explanation.base_values[0], 3))

# ---------------------------------------------------------------- [7] shap_top
pushes = pd.DataFrame(explanation.values, columns=X.columns)
print(pushes.abs().mean().nlargest(10).round(3))   # typical size of the push

# ---------------------------------------------------------------- [8] beeswarm
shap.plots.beeswarm(explanation, max_display=10)   # one dot per booking

# ---------------------------------------------------------------- [9] rates
# Check each driver against the raw data: share canceled in each group
days = pd.cut(df["lead_time"], [-1, 7, 30, 90, 180, 800],
              labels=["0-7", "8-30", "31-90", "91-180", "181+"])
print(df.groupby(days, observed=True)["is_canceled"].mean().round(3))

requests = df["total_of_special_requests"].clip(upper=3)   # 3 means 3 or more
print(df.groupby(requests)["is_canceled"].mean().round(3))

parking = df["required_car_parking_spaces"] > 0            # asked for parking?
print(df.groupby(parking)["is_canceled"].agg(["size", "mean"]).round(3))

# ---------------------------------------------------------------- [10] deposit
print(df.groupby("deposit_type")["is_canceled"].agg(["size", "mean"]).round(3))

# ---------------------------------------------------------------- [11] who
# Who are the Non Refund bookings?
df["non_refund"] = df["deposit_type"] == "Non Refund"
df["from_portugal"] = df["country"] == "PRT"
df["group_booking"] = df["market_segment"] == "Groups"
df["no_requests"] = df["total_of_special_requests"] == 0
who = df.groupby("non_refund")[["from_portugal", "group_booking",
                                "no_requests", "lead_time"]]
print(who.mean().round(2).T)               # lead_time = average days ahead

# ---------------------------------------------------------------- [12] agents
non_refund = df[df["non_refund"]]
print(len(non_refund), "Non Refund bookings")
print(non_refund["agent"].value_counts().head(5))   # which agents made them?

# ---------------------------------------------------------------- [13] country
print(df.groupby("from_portugal")["is_canceled"].mean().round(3))

# Same comparison, but leave out the Non Refund bookings
others = df[df["deposit_type"] != "Non Refund"]
print(others.groupby("from_portugal")["is_canceled"].mean().round(3))

# ---------------------------------------------------------------- [14] years
# Guests who canceled before: does the pattern stay the same every year?
before = df["previous_cancellations"] > 0
table = df.pivot_table(index=before, columns="arrival_date_year",
                       values="is_canceled", aggfunc="mean")
print(table.round(2))

# ---------------------------------------------------------------- [15] one_booking
# Explain one 2017 booking: the first one in our sample
print(df.loc[sample.index[0], ["lead_time", "deposit_type", "country",
                               "market_segment", "total_of_special_requests"]])
print("Forest says:", round(forest.predict_proba(sample.iloc[[0]])[0, 1], 3))
shap.plots.waterfall(explanation[0], max_display=9)

# ---------------------------------------------------------------- [16] action
# A group the manager can act on: online agencies, booked 90+ days ahead
df["online_agency"] = df["market_segment"] == "Online TA"
df["booked_early"] = df["lead_time"] > 90
no_deposit = df[df["deposit_type"] == "No Deposit"]   # the normal bookings
groups = no_deposit.groupby(["online_agency", "booked_early"])["is_canceled"]
print(groups.agg(["size", "mean"]).round(3))
