"""Lecture 6: explaining tree models on the Ames house prices.
Put AmesHousing_clean.csv in the same folder, then run:
    python L6_explain_examples.py
You need the shap package (pip install shap). Charts open in a window: close each one to continue.
"""

# ---------------------------------------------------------------- [1] load
import pandas as pd

df = pd.read_csv("AmesHousing_clean.csv")   # same file as Lectures 3-5

# Same features as Lecture 4
not_features = ["SalePrice", "logp", "Order", "PID", "Yr Sold",
                "Mo Sold", "Sale Type", "Sale Condition"]
X = df.drop(columns=not_features)
X = pd.get_dummies(X, drop_first=True, dtype=float)  # words -> 0/1 columns
X = X.fillna(0)                                      # fill missing values
y = df["SalePrice"]

# Learn from 2006-2008, check on 2009
train = df["Yr Sold"] <= 2008
valid = df["Yr Sold"] == 2009
X_train, y_train = X[train], y[train]
X_valid, y_valid = X[valid], y[valid]
print(len(X_train), "training houses,", len(X_valid), "validation houses")

# ---------------------------------------------------------------- [2] forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error as rmse

forest = RandomForestRegressor(n_estimators=500,   # 500 trees
                               max_features=0.33,  # each split sees 1/3
                               random_state=0,     # same result every run
                               n_jobs=-1)          # use all CPU cores
forest.fit(X_train, y_train)                       # learn from 2006-2008

start_error = rmse(y_valid, forest.predict(X_valid))
print("Error on 2009 houses:", round(start_error))

# ---------------------------------------------------------------- [3] shap_one
import shap

explainer = shap.TreeExplainer(forest)    # a fast explainer for tree models
house = X_valid.loc[[352]]                # our house, as a one-row table
explanation = explainer(house)            # SHAP values for this house

average = explanation.base_values[0]      # where every house starts
pushes = explanation.values[0]            # one push for each feature

print("Starting point:", round(average))
print("All pushes:    ", round(pushes.sum()))
print("Final price:   ", round(average + pushes.sum()))
print("Forest says:   ", round(forest.predict(house)[0]))

# ---------------------------------------------------------------- [4] waterfall
shap.plots.waterfall(explanation[0], max_display=9)   # draw this house

# ---------------------------------------------------------------- [5] shap_top
pushes = pd.Series(pushes, index=X.columns)   # give each push its feature name
biggest = pushes.abs().nlargest(8).index      # the 8 biggest, up or down
print(pushes[biggest].round())                # in dollars

# ---------------------------------------------------------------- [6] shap_cheap
cheap = explainer(X_valid.loc[[401]])         # a much cheaper house
cheap_pushes = pd.Series(cheap.values[0], index=X.columns)
biggest = cheap_pushes.abs().nlargest(4).index   # the 4 biggest
print(cheap_pushes[biggest].round())

# ---------------------------------------------------------------- [7] shap_all
all_houses = explainer(X_valid)   # every 2009 house (takes about 2 minutes)
print(all_houses.shape)           # (number of houses, number of features)

# ---------------------------------------------------------------- [8] beeswarm
shap.plots.beeswarm(all_houses, max_display=10)   # one dot per house

# ---------------------------------------------------------------- [9] mean_abs
# Typical size of the push, up or down, across all 2009 houses
average_push = pd.Series(abs(all_houses.values).mean(axis=0), index=X.columns)
print(average_push.nlargest(8).round())       # in dollars

# ---------------------------------------------------------------- [10] shuffle_one
shuffled = X_valid.copy()                  # a copy: the real data stays safe
mixed = X_valid["Overall Qual"].sample(frac=1, random_state=0)   # random order
shuffled["Overall Qual"] = mixed.values    # the values land on the wrong houses

new_error = rmse(y_valid, forest.predict(shuffled))   # same forest, no retraining
print("Error before:   ", round(start_error))
print("Error after:    ", round(new_error))
print("Error went up by", round(new_error - start_error))

# ---------------------------------------------------------------- [11] shuffle_many
import numpy as np

def shuffle_test(columns):                # columns = a list of column names
    rises = []
    for seed in range(10):                # shuffle 10 times
        order = np.random.RandomState(seed).permutation(len(X_valid))
        shuffled = X_valid.copy()
        shuffled[columns] = X_valid[columns].values[order]   # same new order
        new_error = rmse(y_valid, forest.predict(shuffled))
        rises.append(new_error - start_error)
    return round(np.mean(rises))          # average rise in the error

print("Quality:      ", shuffle_test(["Overall Qual"]))
print("Living area:  ", shuffle_test(["Gr Liv Area"]))
print("Garage Cars:  ", shuffle_test(["Garage Cars"]))
print("Garage Area:  ", shuffle_test(["Garage Area"]))
print("Both together:", shuffle_test(["Garage Cars", "Garage Area"]))

# ---------------------------------------------------------------- [12] averages
# Used in the last part of the lecture
high_quality = X_train["Overall Qual"] >= 8   # True/False for each house
large_area = X_train["Gr Liv Area"] >= 2000   # True/False for each house

print("All houses:  ", round(y_train.mean()))
print("High quality:", round(y_train[high_quality].mean()))
print("Large area:  ", round(y_train[large_area].mean()))
print("Both:        ", round(y_train[high_quality & large_area].mean()))

