import json
import os
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")

#  load the numbers
with open(os.path.join(RESULTS_DIR, "numbers.json"), "r") as f:
    N = json.load(f)

#  extract the tuition gap data
up_to_date = N["tuition_gap"]["up_to_date_1"]
not_up_to_date = N["tuition_gap"]["up_to_date_0"]

#  set up the chart
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6, 5))

labels = ["Tuition up to date", "Not up to date"]
values = [up_to_date, not_up_to_date]
colors = ["#13807A", "#C0392B"] # Teal for good, Red for bad

bars = ax.bar(labels, values, color=colors, width=0.55)

#  add the percentage labels
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}",
            ha="center", fontsize=16, fontweight="bold", color="#1B2A4A")

ax.set_ylabel("Dropout Rate", fontsize=14)
ax.set_ylim(0, max(values) * 1.25)
ax.set_title("Dropout Rate by Tuition Status", fontsize=16, fontweight="bold", pad=15)

#  save figure
plt.savefig(os.path.join(FIGURES_DIR, "fig_slide3_tuition.png"),
            dpi=200, bbox_inches="tight")
print("Saved figures/fig_slide3_tuition.png")