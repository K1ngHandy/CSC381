"""
Step 2 of 2 - check the model, then use SHAP to find the drivers of MPG.

  1. Train on 1984-2019 and test on 2020-2025 (the same split as Report 2).
  2. Refit gradient boosting on all vehicles and compute SHAP values.
  3. Make the tables and figures in our deck.

Run from the project folder, after step 1:  python src/02_driver_analysis.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.family": "Calibri", "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False})
NAVY, TEAL, GREEN, RED, GREY = "#1B2A4A", "#13807A", "#2E8B57", "#C0392B", "#8A94A6"

df = pd.read_csv(os.path.join("data", "clean.csv"))
X = pd.get_dummies(df.drop(columns=["make", "model", "comb08"]), columns=["kind"],
                   drop_first=True, dtype=int)
y = df["comb08"]

NAMES = {"year": "Model year", "displ": "Engine size (L)", "cylinders": "Cylinders",
         "all_wheel": "All-wheel drive", "manual": "Manual transmission",
         "turbo": "Turbo", "hybrid": "Hybrid", "kind_Pickup": "Pickup",
         "kind_SUV": "SUV", "kind_Van/other": "Van/other", "kind": "Body type"}
KIND_COLS = ["kind_Pickup", "kind_SUV", "kind_Van/other"]
N = {}   # numbers we quote in the deck


def save(fig, name):
    fig.savefig(os.path.join("figures", name), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 1. model check
train = df["year"] <= 2019
X_tr, X_te, y_tr, y_te = X[train], X[~train], y[train], y[~train]
models = {
    "Baseline (always guess the average)": None,
    "Linear regression": LinearRegression(),
    "Random forest (Report 2)": RandomForestRegressor(n_estimators=200, random_state=0,
                                                      n_jobs=-1),
    "Gradient boosting (ours)": GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                                          random_state=0),
}
rows = []
for name, model in models.items():
    if model is None:
        pred = np.full(len(y_te), y_tr.mean())
    else:
        pred = model.fit(X_tr, y_tr).predict(X_te)
    rows.append({"model": name, "MAE": round(mean_absolute_error(y_te, pred), 2),
                 "R2": round(r2_score(y_te, pred), 2)})
pd.DataFrame(rows).to_csv(os.path.join("results", "model_scores.csv"), index=False)
print(pd.DataFrame(rows).to_string(index=False))
N["scores"] = rows
N["n_train"], N["n_test"], N["n_all"] = int(train.sum()), int((~train).sum()), len(df)

# ---------------------------------------------------------------- 2. SHAP
# The test above says the model is good enough, so we refit it on all years
# and explain it.
model = GradientBoostingRegressor(n_estimators=300, max_depth=4, random_state=0)
model.fit(X, y)
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X)
base = float(np.ravel(explainer.expected_value)[0])
N["base"] = round(base, 1)

S = pd.DataFrame(sv, columns=X.columns, index=X.index)
# Body type is three 0/1 columns. Adding their SHAP values gives one number
# for body type.
S_body = S.drop(columns=KIND_COLS)
S_body["kind"] = S[KIND_COLS].sum(axis=1)

# ---------------------------------------------------------------- 3. tables
# Driver ranking: mean absolute SHAP value of each feature.
ranking = S_body.abs().mean().sort_values(ascending=False)
N["ranking"] = [{"col": c, "name": NAMES[c], "mean_abs_shap": round(v, 2)}
                for c, v in ranking.items()]
pd.DataFrame(N["ranking"]).to_csv(os.path.join("results", "driver_ranking.csv"),
                                  index=False)
print(ranking.round(2).to_string())

# Engine size: average SHAP value by engine size group.
groups = pd.cut(df["displ"], [0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 9],
                labels=["2.0 or less", "2.1-2.5", "2.6-3.0", "3.1-3.5", "3.6-4.0",
                        "4.1-5.0", "over 5.0"])
N["engine_groups"] = S["displ"].groupby(groups, observed=True).mean().round(1).to_dict()

# Hybrid and turbo: average SHAP value for vehicles that have them.
for col in ["hybrid", "turbo"]:
    N[col + "_yes"] = round(float(S.loc[X[col] == 1, col].mean()), 1)
    N[col + "_share"] = round(float(X[col].mean()) * 100, 1)

# Report 1 hypothesis. For each body type we take the average SHAP value of
# every feature and subtract the average for cars. The differences add up to
# the gap between the average predictions.
N["avg_mpg"] = df.groupby("kind")["comb08"].mean().round(1).to_dict()
N["avg_displ"] = df.groupby("kind")["displ"].mean().round(1).to_dict()
by_body = S_body.groupby(df["kind"]).mean()
N["gaps"] = {}
for k in ["Pickup", "SUV"]:
    diff = (by_body.loc[k] - by_body.loc["Car"]).sort_values()
    N["gaps"][k] = {"total": round(float(diff.sum()), 1),
                    "parts": {NAMES[c]: round(float(v), 1) for c, v in diff.items()}}
print(json.dumps(N["gaps"], indent=1))

# One vehicle for the waterfall chart.
car = df.index[(df["year"] == 2025) & (df["model"] == "RAV4 Hybrid AWD")][0]
N["car"] = {"name": "2025 Toyota RAV4 Hybrid AWD", "actual": int(y[car]),
            "pred": round(float(model.predict(X.loc[[car]])[0]), 1),
            "shap": {NAMES[c]: round(float(v), 1) for c, v in S.loc[car].items()}}

# ---------------------------------------------------------------- figures
names = [NAMES[c] for c in X.columns]

# fig1: average MPG by body type (what Report 1 showed)
order = ["Car", "SUV", "Pickup", "Van/other"]
avg = N["avg_mpg"]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
bars = ax.bar(order, [avg[k] for k in order], color=[TEAL, GREY, GREY, GREY], width=0.6)
for b, k in zip(bars, order):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{avg[k]:.1f}",
            ha="center", fontsize=14, fontweight="bold", color=NAVY)
ax.set_ylabel("Average combined MPG")
ax.set_ylim(0, 26)
save(fig, "fig1_mpg_by_body.png")

# fig2: SHAP waterfall for one vehicle. We changed SHAP's colors to green
# (raises MPG) and red (lowers MPG) to match the rest of our deck.
exp = shap.Explanation(values=sv[X.index.get_loc(car)], base_values=base,
                       data=X.loc[car].values, feature_names=names)
shap.plots.waterfall(exp, max_display=8, show=False)
fig = plt.gcf()


def recolor(c):
    if c[0] > 0.8 and c[2] < 0.6:
        return GREEN
    if c[2] > 0.8 and c[0] < 0.3:
        return RED
    return None


for a in fig.axes:
    for p in a.patches:
        new = recolor(matplotlib.colors.to_rgb(p.get_facecolor()))
        if new:
            p.set_facecolor(new)
            p.set_edgecolor(new)
    for t in a.texts:
        new = recolor(matplotlib.colors.to_rgb(t.get_color()))
        if new:
            t.set_color(new)
fig.set_size_inches(7.6, 4.6)
save(fig, "fig2_waterfall_rav4.png")

# fig3: beeswarm for 4,000 random vehicles
rng = np.random.default_rng(0)
pick = rng.choice(len(X), 4000, replace=False)
exp_all = shap.Explanation(values=sv[pick], base_values=np.full(len(pick), base),
                           data=X.values[pick], feature_names=names)
shap.plots.beeswarm(exp_all, max_display=10, show=False)
fig = plt.gcf()
fig.set_size_inches(7.6, 4.8)
save(fig, "fig3_beeswarm.png")

# fig4: engine size against its SHAP value
fig, ax = plt.subplots(figsize=(6.6, 4.3))
ax.scatter(X["displ"].values[pick] + rng.uniform(-0.05, 0.05, len(pick)),
           S["displ"].values[pick], s=7, alpha=0.25, color=TEAL)
mean_line = S["displ"].groupby(X["displ"].round(1)).mean()
counts = X["displ"].round(1).value_counts()
mean_line = mean_line[counts.reindex(mean_line.index) >= 30]
ax.plot(mean_line.index, mean_line.values, color=NAVY, linewidth=2.5, label="average")
ax.axhline(0, color="#333333", linewidth=1)
ax.set_xlabel("Engine size (liters)")
ax.set_ylabel("SHAP value of engine size (MPG)")
ax.legend(frameon=False, loc="upper right")
save(fig, "fig4_engine.png")

# fig5: SHAP values of hybrid and turbo, for no and yes
fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.0), sharey=True)
for ax, col in zip(axes, ["hybrid", "turbo"]):
    data = [S.loc[X[col] == 0, col].values, S.loc[X[col] == 1, col].values]
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for body, c in zip(parts["bodies"], [GREY, GREEN if col == "hybrid" else RED]):
        body.set_facecolor(c)
        body.set_alpha(0.7)
    for i, d in enumerate(data, 1):
        ax.text(i, d.mean(), f"{d.mean():+.1f}", ha="center", va="bottom",
                fontsize=14, fontweight="bold", color=NAVY)
    ax.set_xticks([1, 2], ["no", "yes"])
    ax.set_title(NAMES[col], fontsize=15, color=NAVY)
    ax.axhline(0, color="#333333", linewidth=1)
axes[0].set_ylabel("SHAP value (MPG)")
save(fig, "fig5_hybrid_turbo.png")

# fig6: where the pickup and SUV gaps come from
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharex=True,
                         gridspec_kw={"wspace": 0.75})
for ax, k in zip(axes, ["Pickup", "SUV"]):
    parts = N["gaps"][k]["parts"]
    keep = {n: v for n, v in parts.items() if abs(v) >= 0.15}
    rest = round(N["gaps"][k]["total"] - sum(keep.values()), 1)
    if abs(rest) >= 0.05:
        keep["other features"] = rest
    labels = list(keep)[::-1]
    vals = [keep[n] for n in labels]
    ax.barh(labels, vals, color=[GREEN if v > 0 else RED for v in vals], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + (0.06 if v > 0 else -0.06), i, f"{v:+.1f}", va="center",
                ha="left" if v > 0 else "right", fontsize=12)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title(f"{k} vs car: {N['gaps'][k]['total']:+.1f} MPG", fontsize=15,
                 color=NAVY)
    ax.set_xlabel("Difference in mean SHAP value (MPG)")
ax.set_xlim(-4.6, 1.8)
save(fig, "fig6_gap_parts.png")

with open(os.path.join("results", "numbers.json"), "w") as f:
    json.dump(N, f, indent=1, default=float)
print("Saved results/ and figures/")
