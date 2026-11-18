# What drives a car's fuel economy?

CSC 381/576 · Capstone example · Team 0 (a made-up team)

This is an example project. It is not the best possible analysis.

We use Student A's dataset from Report 1 and Report 2, the U.S. EPA fuel economy
file. In Report 2 we showed that a model can predict the combined MPG of a
vehicle from its spec sheet. Here we use SHAP values to see which features
drive the predictions, and we check the hypothesis from Report 1 that body
type (pickup, SUV) is the reason for low MPG.

## Data

U.S. EPA, https://www.fueleconomy.gov/feg/epadata/vehicles.csv (about 21 MB,
not stored in this folder; step 1 downloads it into `data/`).
One row is one vehicle configuration in one model year. We keep 46,185
gasoline vehicles from 1984 to 2025.

## How to run

```
pip install -r requirements.txt
python src/01_prepare_data.py      # download, filter, build features -> data/clean.csv
python src/02_driver_analysis.py   # model check, SHAP, tables, figures
```

Run both from this folder. It takes about a minute.

## Files

| File | What it does |
|---|---|
| `src/01_prepare_data.py` | Builds the same eight features as Report 2, and keeps make and model as labels |
| `src/02_driver_analysis.py` | Checks the model on 2020-2025, refits on all years, computes SHAP values, saves tables and figures |
| `results/model_scores.csv` | Model check (slide 5) |
| `results/driver_ranking.csv` | Feature ranking by mean absolute SHAP value (slide 7) |
| `results/numbers.json` | Numbers quoted in the deck |
| `figures/` | Charts in the deck |

## Who did what

| Member | Work |
|---|---|
| Student A | Dataset owner, data preparation, body type analysis |
| Student B | Model check, SHAP values, engine size analysis |
| Student C | Hybrid and turbo analysis, limitations |

## AI note

We used AI to help write parts of the code and the text. Every number in the
deck comes from running the two scripts.
