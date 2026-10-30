"""Lecture 4: decision trees and random forests on the Ames house prices.
Put AmesHousing_clean.csv in the same folder, then run:
    python L4_tree_examples.py
"""

# ---------------------------------------------------------------- [1] load
import pandas as pd

df = pd.read_csv("AmesHousing_clean.csv")   # same file as Lecture 3

# Columns we can't use: the price itself, IDs, and sale details
not_features = ["SalePrice", "logp", "Order", "PID", "Yr Sold",
                "Mo Sold", "Sale Type", "Sale Condition"]
X = df.drop(columns=not_features)
X = pd.get_dummies(X, drop_first=True, dtype=float)  # words -> 0/1 columns
X = X.fillna(0)                                      # fill missing values
y = df["SalePrice"]                                  # target: sale price

print(X.shape)        # (number of houses, number of features)

# ---------------------------------------------------------------- [2] split
from sklearn.metrics import root_mean_squared_error as rmse

# Split by time: learn from 2006-2008, check on 2009
train = df["Yr Sold"] <= 2008
valid = df["Yr Sold"] == 2009

X_train, y_train = X[train], y[train]
X_valid, y_valid = X[valid], y[valid]

print(len(X_train), "training houses")
print(len(X_valid), "validation houses")

# ---------------------------------------------------------------- [3] tree_clf
from sklearn.tree import DecisionTreeClassifier, export_text

# Yes/no target: did the house sell for more than $160K?
expensive = y > 160000

clf = DecisionTreeClassifier(max_depth=2,      # at most 2 questions
                             random_state=0)   # same result every run
clf.fit(X_train, expensive[train])             # learn the questions

print(export_text(clf, feature_names=list(X.columns)))

# ---------------------------------------------------------------- [4] clf_acc
right = clf.predict(X_valid) == expensive[valid]   # True = correct guess
print("Validation accuracy:", round(right.mean(), 3))

# ---------------------------------------------------------------- [5] tree_small
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=2,      # at most 2 questions
                             random_state=0)   # same result every run
tree.fit(X_train, y_train)                     # learn the questions

# Print the tree as if-else rules
print(export_text(tree, feature_names=list(X.columns)))

# ---------------------------------------------------------------- [6] units
cols = ["Gr Liv Area", "Overall Qual", "Garage Cars"]
feet = X_train[cols]                              # living area in sq ft
meters = feet.copy()
meters["Gr Liv Area"] = feet["Gr Liv Area"] * 0.0929   # now in m2

tree_ft = DecisionTreeRegressor(max_depth=5, random_state=0)
tree_m = DecisionTreeRegressor(max_depth=5, random_state=0)
tree_ft.fit(feet, y_train)
tree_m.fit(meters, y_train)

same = (tree_ft.predict(feet) == tree_m.predict(meters)).all()
print("Same predictions?", same)

# ---------------------------------------------------------------- [7] depth
for depth in [2, 4, 6, 10, None]:              # None = no limit
    tree = DecisionTreeRegressor(max_depth=depth, random_state=0)
    tree.fit(X_train, y_train)

    train_error = rmse(y_train, tree.predict(X_train))
    valid_error = rmse(y_valid, tree.predict(X_valid))
    print("max_depth", depth, "| train", round(train_error),
          "| valid", round(valid_error))

# ---------------------------------------------------------------- [8] leaf
# Another way to stop the tree: each leaf needs at least 10 houses
tree = DecisionTreeRegressor(min_samples_leaf=10, random_state=0)
tree.fit(X_train, y_train)
print("train", round(rmse(y_train, tree.predict(X_train))),
      "| valid", round(rmse(y_valid, tree.predict(X_valid))))

# ---------------------------------------------------------------- [9] mansion
from sklearn.linear_model import LinearRegression

area = X_train[["Gr Liv Area"]]                  # one feature only
tree = DecisionTreeRegressor(max_depth=3, random_state=0)
line = LinearRegression()
tree.fit(area, y_train)
line.fit(area, y_train)

# Three new houses: 3,000 / 3,820 / 6,000 sq ft
new = pd.DataFrame({"Gr Liv Area": [3000, 3820, 6000]})
print("Tree:", tree.predict(new).round(-3))     # round to $1,000
print("Line:", line.predict(new).round(-3))

# ---------------------------------------------------------------- [10] halves
half1 = X_train.sample(frac=0.5, random_state=1)  # a random half
half2 = X_train.drop(half1.index)                  # the other half

tree1 = DecisionTreeRegressor(random_state=0)      # no depth limit
tree2 = DecisionTreeRegressor(random_state=0)
tree1.fit(half1, y_train[half1.index])
tree2.fit(half2, y_train[half2.index])

# Same 2009 houses, two trees: how far apart are their prices?
gap = abs(tree1.predict(X_valid) - tree2.predict(X_valid))
print("Average disagreement: $", round(gap.mean()))

# ---------------------------------------------------------------- [11] forest
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=500,  # 500 trees
                               max_features=0.33, # each split sees 1/3 of features
                               oob_score=True,    # also do the OOB check
                               random_state=0)
forest.fit(X_train, y_train)

# ---------------------------------------------------------------- [12] forest_eval
print("train:", round(rmse(y_train, forest.predict(X_train))))
print("OOB:  ", round(rmse(y_train, forest.oob_prediction_)))  # free check
print("valid:", round(rmse(y_valid, forest.predict(X_valid))))

# ---------------------------------------------------------------- [13] maxfeat
for share in [1.0, 0.33]:          # 1.0 = all features = plain bagging
    f = RandomForestRegressor(n_estimators=500, max_features=share,
                              random_state=0)
    f.fit(X_train, y_train)
    print("max_features", share, "| valid", round(rmse(y_valid, f.predict(X_valid))))

# ---------------------------------------------------------------- [14] ntrees
for n in [1, 10, 100, 500]:
    f = RandomForestRegressor(n_estimators=n, max_features=0.33,
                              random_state=0)
    f.fit(X_train, y_train)
    print(n, "trees | valid", round(rmse(y_valid, f.predict(X_valid))))

# ---------------------------------------------------------------- [15] importance
importance = pd.Series(forest.feature_importances_, index=X.columns)
print(importance.nlargest(8).round(3))    # the 8 most important features
