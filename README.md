# Clinical Trials Analysis

Python workspace for exploring clinical-trial records and condition queries.

The supplied CSV remains in the project root. Keep source data out of version
control; use `data/raw/` for input datasets and `data/processed/` for derived
outputs when the analysis pipeline is added.

## Clean the supplied data

Install dependencies with `pip install -r requirements.txt`, then run:

```sh
python src/clinical_trials/clean_data.py
```

The cleaner writes `data/processed/clinical_trials_cleaned.csv` and an
aggregate-only report to `reports/cleaning_report.json`. It preserves the
source columns and records, converts empty cells to missing values, standardizes
trial metadata codes, and adds numeric `minimum_age_years` and
`maximum_age_years` columns. The original age strings remain available.

## Project layout

```text
data/
  raw/             Input datasets
  processed/       Cleaned or derived datasets
notebooks/         Exploratory analysis
reports/figures/   Generated charts
src/clinical_trials/  Reusable analysis code
tests/             Automated tests
```

Activate the virtual environment with `source .venv/bin/activate`.