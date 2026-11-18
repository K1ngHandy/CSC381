# What drives student dropout risk?

CSC 576 · Capstone Project · Team [Your Team Name]

This is our capstone project. We used the Kaggle "Student Academic Performance & Dropout Risk" dataset. In our earlier reports, we showed that we can predict whether a student will drop out based on their academic, financial, and engagement data. Here, we use SHAP values to see exactly which features drive the model's predictions, and we check our hypothesis that financial distress is the main reason for dropout.

## Data

Kaggle, Student Academic Performance & Dropout Risk (not stored in this folder; see the Kaggle link in our report).
One row is one student. We use 100,000 records with 32 features. 
Target: `dropout` (1 = Dropped out, 0 = Enrolled/Graduated). The positive rate is 32%.

## How to run

pip install -r requirements.txt
python src/01_prepare_data.py # drop ID, clean data -> data/clean.csv
python src/02_driver_analysis.py # model check, SHAP, tables, figures

Run both from this folder. It takes about a minute.

## Files

| File | What it does |
|---|---|
| `src/01_prepare_data.py` | Cleans the Kaggle CSV, drops `person_id`, and saves `data/clean.csv` |
| `src/02_driver_analysis.py` | Trains models on 75% of data, checks metrics (Accuracy/F1/AUC), refits on all data, computes SHAP values, saves tables and figures |
| `results/model_scores.csv` | Model check (Slide 6) |
| `results/driver_ranking.csv` | Feature ranking by mean absolute SHAP value (Slide 8) |
| `results/numbers.json` | Numbers quoted in the deck |
| `figures/` | Charts in the deck |
