#  step 2 - model check, SHAP driver analysis, tables, & figures

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False})
NAVY, TEAL, GREEN, RED, GREY = "#1B2A4A", "#13807A", "#2E8B57", "#C0392B", "#8A94A6"

df = pd.read_csv(os.path.join("data", "clean.csv"))
y = df["dropout"]
X = pd.get_dummies(df.drop(columns=["dropout"]), drop_first=True, dtype=int)

N = {}

#  model check
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

models = {
    "Baseline (majority class)": None,
    "Logistic regression": LogisticRegression(max_iter=5000),
    "Random forest": RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
    "Gradient boosting (ours)": GradientBoostingClassifier(
        n_estimators=300, max_depth=3, random_state=0
    ),
}

rows = []
for name, model in models.items():
    if model is None:
        pred = np.zeros(len(y_test))
        prob = np.full(len(y_test), y_train.mean())
    else:
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
    rows.append({
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 3),
        "f1": round(f1_score(y_test, pred), 3),
        "auc": round(roc_auc_score(y_test, prob), 3),
    })

pd.DataFrame(rows).to_csv(os.path.join("results", "model_scores.csv"), index=False)
print(pd.DataFrame(rows).to_string(index=False))
N["scores"] = rows
N["n_train"], N["n_test"], N["n_all"] = len(X_train), len(X_test), len(X)
N["base_rate"] = round(float(y.mean()), 3)

# SHAP

#  refit on all data and explain the full model.
model = GradientBoostingClassifier(n_estimators=300, max_depth= 3, random_state=0)
model.fit(X, y)
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X)

#  for binary classifiers SHAP may return a list or a single array.
if isinstance(sv, list):
    sv = sv[1]  # positive class = dropout

base = float(np.ravel(explainer.expected_value)[-1]) if isinstance(
    explainer.expected_value, (list, np.ndarray)
) else float(explainer.expected_value)
N["base_logodds"] = round(base, 3)

S = pd.DataFrame(sv, columns=X.columns, index=X.index)

#  ranking
ranking = S.abs().mean().sort_values(ascending=False)
N["ranking"] = [{"col": c, "name": c, "mean_abs_shap": round(v, 3)}
                for c, v in ranking.items()]
pd.DataFrame(N["ranking"]).to_csv(os.path.join("results", "driver_ranking.csv"),
                                  index=False)
print("\nTop drivers by mean |SHAP|:")
print(ranking.head(12).round(3).to_string())

#  evidence numbers

#  tuition gap: dropout rate for tuition_up_to_date = 1 vs 0
N["tuition_gap"] = {
    "up_to_date_1": round(float(df.loc[df["tuition_up_to_date"] == 1, "dropout"].mean()), 3),
    "up_to_date_0": round(float(df.loc[df["tuition_up_to_date"] == 0, "dropout"].mean()), 3),
}

#  part-time job gap
N["part_time_gap"] = {
    "job_1": round(float(df.loc[df["part_time_job"] == 1, "dropout"].mean()), 3),
    "job_0": round(float(df.loc[df["part_time_job"] == 0, "dropout"].mean()), 3),
}

#  attendance bins vs dropout rate
bins = [0, 60, 75, 90, 101]
labels = ["<60%", "60-75%", "75-90%", "90-100%"]
df["attendance_bin"] = pd.cut(df["attendance_rate"], bins, labels=labels, right=False)
N["attendance_dropout"] = (
    df.groupby("attendance_bin", observed=True)["dropout"].mean().round(3).to_dict()
)

#  gpa semester 1 bins
gpa_bins = [0, 1.5, 2.0, 2.5, 3.0, 4.0]
gpa_labels = ["<1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0-4.0"]
df["gpa1_bin"] = pd.cut(df["gpa_semester_1"], gpa_bins, labels=gpa_labels, right=False)
N["gpa1_dropout"] = (
    df.groupby("gpa1_bin", observed=True)["dropout"].mean().round(3).to_dict()
)

#  stress level vs dropout
N["stress_dropout"] = (
    df.groupby("stress_level")["dropout"].mean().round(3).to_dict()
)

#  figures

#  fig1 - dropout rate by tuition status
fig, ax = plt.subplots(figsize=(5.6, 4.0))
vals = [N["tuition_gap"]["up_to_date_1"], N["tuition_gap"]["up_to_date_0"]]
bars = ax.bar(["Tuition up to date", "Not up to date"], vals,
              color=[TEAL, RED], width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}",
            ha="center", fontsize=14, fontweight="bold", color=NAVY)
ax.set_ylabel("Dropout rate")
ax.set_ylim(0, max(vals) * 1.25)
save = lambda f, n: (f.savefig(os.path.join("figures", n), dpi=200,
                                bbox_inches="tight"), plt.close(f))
save(fig, "fig1_tuition_dropout.png")

#  fig2 - dropout rate by attendance bin
fig, ax = plt.subplots(figsize=(6.4, 4.0))
labels = list(N["attendance_dropout"].keys())
vals = list(N["attendance_dropout"].values())
bars = ax.bar(labels, vals, color=[RED if v > 0.35 else GREY for v in vals], width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}",
            ha="center", fontsize=13, fontweight="bold", color=NAVY)
ax.set_ylabel("Dropout rate")
ax.set_xlabel("Attendance rate")
ax.set_ylim(0, max(vals) * 1.3)
save(fig, "fig2_attendance_dropout.png")

#  fig3 - SHAP beeswarm
shap.summary_plot(sv, X, show=False, max_display=12)
fig = plt.gcf()
fig.set_size_inches(8.0, 5.2)
save(fig, "fig3_beeswarm.png")

#  fig4 - SHAP waterfall for one high-risk student
high_risk = model.predict_proba(X)[:, 1].argmax()
exp = shap.Explanation(
    values=sv[high_risk],
    base_values=base,
    data=X.iloc[high_risk].values,
    feature_names=X.columns,
)
shap.plots.waterfall(exp, max_display=10, show=False)
fig = plt.gcf()
fig.set_size_inches(8.0, 5.0)
save(fig, "fig4_waterfall_high_risk.png")

#  fig5 - dropout rate by GPA semester 1 bin
fig, ax = plt.subplots(figsize=(6.4, 4.0))
labels = list(N["gpa1_dropout"].keys())
vals = list(N["gpa1_dropout"].values())
bars = ax.bar(labels, vals, color=[RED if v > 0.35 else GREY for v in vals], width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}",
            ha="center", fontsize=12, fontweight="bold", color=NAVY)
ax.set_ylabel("Dropout rate")
ax.set_xlabel("GPA semester 1")
ax.set_ylim(0, max(vals) * 1.3)
save(fig, "fig5_gpa_dropout.png")

with open(os.path.join("results", "numbers.json"), "w") as f:
    json.dump(N, f, indent=1, default=str)
print("\nSaved results/ and figures/")