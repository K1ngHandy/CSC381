"""Lecture 5: boosting. Part 1 builds boosting by hand on 8 houses; Part 2 uses the Ames data.
Put AmesHousing_clean.csv in the same folder, then run:
    python L5_boosting_examples.py
"""

# ---------------------------------------------------------------- [1] hand_data
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# The 8 houses from Lecture 4
houses = pd.DataFrame({"quality":   [6, 8, 5, 8, 6, 7, 5, 7],
                       "fireplace": [1, 1, 0, 1, 0, 1, 0, 0],   # 1 = yes
                       "price":     [140200, 260000, 111000, 280000,
                                     127500, 255000, 112000, 190000]},
                      index=list("ABCDEFGH"))
X8 = houses[["quality", "fireplace"]]
y8 = houses["price"]
print("Step 0 guess (the average):", round(y8.mean()))

# ---------------------------------------------------------------- [2] hand_loop
guess = y8.mean()                          # step 0: the average price

for step in [1, 2, 3]:
    leftover = y8 - guess                  # what is still wrong
    tree = DecisionTreeRegressor(max_depth=1)   # a tiny tree: 1 question
    tree.fit(X8, leftover)                 # it learns the leftover
    guess = guess + tree.predict(X8)       # add its fix to the guess

    error = abs(y8 - guess).mean()         # typical error now
    print("after tree", step, "| typical error:", round(error))

# ---------------------------------------------------------------- [3] load
import numpy as np
from sklearn.metrics import root_mean_squared_error as rmse

df = pd.read_csv("AmesHousing_clean.csv")   # same file and features as Lecture 4
not_features = ["SalePrice", "logp", "Order", "PID", "Yr Sold",
                "Mo Sold", "Sale Type", "Sale Condition"]
X = df.drop(columns=not_features)
X = pd.get_dummies(X, drop_first=True, dtype=float)   # words -> 0/1 columns
X = X.fillna(0)
y = df["SalePrice"]

train = df["Yr Sold"] <= 2008                # learn from 2006-2008
valid = df["Yr Sold"] == 2009                # check on 2009
X_train, y_train = X[train], y[train]
X_valid, y_valid = X[valid], y[valid]

# ---------------------------------------------------------------- [4] boost
from sklearn.ensemble import GradientBoostingRegressor

boost = GradientBoostingRegressor(n_estimators=500,    # 500 small trees
                                  learning_rate=0.1,   # use 10% of each fix
                                  max_depth=3,         # each tree: 3 questions
                                  random_state=0)      # same result every run
boost.fit(X_train, y_train)

print("train RMSE:", round(rmse(y_train, boost.predict(X_train))))
print("valid RMSE:", round(rmse(y_valid, boost.predict(X_valid))))

# ---------------------------------------------------------------- [5] lr
for lr in [1.0, 0.1]:
    b = GradientBoostingRegressor(n_estimators=500, learning_rate=lr,
                                  max_depth=3, random_state=0)
    b.fit(X_train, y_train)

    error = rmse(y_valid, b.predict(X_valid))   # error with all 500 trees
    print("learning rate", lr, "| valid RMSE:", round(error))

# ---------------------------------------------------------------- [6] depth
for depth in [1, 2, 3, 5, 8]:
    b = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1,
                                  max_depth=depth, random_state=0)
    b.fit(X_train, y_train)

    error = rmse(y_valid, b.predict(X_valid))
    print("max_depth", depth, "| valid RMSE:", round(error))

# ---------------------------------------------------------------- [7] ridge
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

alphas = np.logspace(-1, 4, 30)                  # penalties to try
ridge = make_pipeline(StandardScaler(),          # scale first (Lecture 3)
                      RidgeCV(alphas=alphas))     # pick the best penalty
ridge.fit(X_train, np.log(y_train))              # Ridge learns log(price)
p_ridge = np.exp(ridge.predict(X_valid))         # back to dollars

# ---------------------------------------------------------------- [8] blend
p_boost = boost.predict(X_valid)
p_avg = (p_ridge + p_boost) / 2               # average the two guesses

print("Ridge:   ", round(rmse(y_valid, p_ridge)))
print("Boosting:", round(rmse(y_valid, p_boost)))
print("Average: ", round(rmse(y_valid, p_avg)))
