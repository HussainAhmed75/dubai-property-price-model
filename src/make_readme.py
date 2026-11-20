"""Renders README.md from results/metrics.json so that every number in the
README is generated, never typed by hand.  Run:  python src/make_readme.py"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = json.load(open(os.path.join(ROOT, "results", "metrics.json"), encoding="utf-8"))
H, D, F, S, T, I, DR, WF = (M["headline"], M["dataset"], M["facts"], M["splits"], M["tuning"],
                            M["interval"], M["drift"], M["walk_forward"])
GH = "https://github.com/HussainAhmed75/dubai-property-price-model"
KAGGLE_DATA = "https://www.kaggle.com/datasets/tomasgrygar/dubai-real-estate-sales-transactions"


def row(name, m, note=""):
    return (f"| {name} | {m['mdape_pct']:.2f}% | {m['within_10pct']:.1f}% | {m['within_20pct']:.1f}% "
            f"| {m['r2_log_psm']:.3f} | {note} |")


order = ["Market baseline", "Comps rule (1 feature)", "Building rule", "Ridge (linear)", "Random Forest",
         "XGBoost", "PerpetualBooster", "LightGBM", "LightGBM (tuned)"]
val_rows = "\n".join(row(n, M["validation"][n]) for n in order if n in M["validation"])
test_order = ["Market baseline", "Comps rule (1 feature)", "Building rule", "Ridge (linear)", "Random Forest",
              "XGBoost", "PerpetualBooster", "LightGBM", "LightGBM (tuned, final)"]
test_rows = "\n".join(row(n, M["test"][n], "**selected on validation**" if n == "LightGBM (tuned, final)" else "reference only")
                      for n in test_order if n in M["test"])
test_rows += "\n" + row("LightGBM retrained monthly (walk-forward)", WF, "same settings, refit before each month")

seg = M["test_by_segment"]
seg_rows = "\n".join(
    f"| {lab} | {v['n']:,.0f} | {v['mdape_model']:.2f}% | {v['mdape_building_rule']:.2f}% |"
    for lab, v in [("Off-plan", seg["reg_type_label"]["Off-Plan"]), ("Ready", seg["reg_type_label"]["Ready"]),
                   ("Unit (apartment)", seg["prop_type"]["Unit"]), ("Villa", seg["prop_type"]["Villa"]),
                   ("Building with ≥3 past sales", seg["has_bldg_history"]["≥3 past sales in building"]),
                   ("New / unknown building", seg["has_bldg_history"]["new / unknown building"])])

clean_rows = "\n".join(f"| {c['step']} | {c['reason']} | {c['rows_removed']:,} | {c['rows_after']:,} |"
                       for c in M["cleaning_log"][1:])

case_rows = "\n".join(
    f"| {c['case']} | {c['date']} | {c['area']} | {c['reg_type']} {c['type']}, {c['size_sqm']} m² | "
    f"{c['actual_aed']:,} | {c['predicted_aed']:,} | {c['error_pct']:+.1f}% | {c['past_sales_in_building']} | "
    f"{', '.join(c['top_shap_features'])} |" for c in M["case_studies"])

misses = [c for c in M["case_studies"] if c["case"] != "hit"]
no_hist = all(c["past_sales_in_building"] == 0 for c in misses)
misses_note = ("The two misses are 99th-percentile errors, not the single worst rows. "
               + ("Both are homes with **no past sales in their building**, so the model falls back on area-level averages — "
                  "and these two homes do not look like their area's average." if no_hist else
                  "See the notebook for the full feature values of each case."))

readme = f"""# What is a Dubai home worth? An honest price model on {D['rows_raw']:,} official sales

Predicting the registered price of Dubai homes from what is known **on the day of the sale**, using residential
sales registered with the Dubai Land Department (DLD). Built to be honest first: no leakage,
a time-based split, tough baselines, and every number in this README generated from
[`results/metrics.json`](results/metrics.json).

**Headline (test period {S['test']['from']} → {S['test']['to']}, never touched during development):**
the tuned LightGBM model gets a **median error of {H['test_mdape_pct']:.2f}%** on the sale price, with
**{H['test_within_10pct']:.1f}%** of {S['test']['n']:,} sales predicted within ±10%. The toughest hand-made rule
(past prices in the same building) gets {H['building_rule_test_mdape_pct']:.2f}% — so the model cuts the
error by **{H['error_reduction_vs_building_rule_pct']:.1f}%**.
**But:** a model frozen at the end of 2024 ages. It beats that rule in only
{DR['frozen_months_better_than_building_rule']} of {DR['n_months']} test months and loses from
{DR['frozen_first_month_worse_than_rule']} on. Retrained every month, it beats the rule in
{DR['walk_forward_months_better_than_building_rule']} of {DR['n_months']} months (median error {WF['mdape_pct']:.2f}%).

![Model comparison](figures/05_model_comparison.png)

## The data
* **Source:** [Dubai Real Estate Data]({KAGGLE_DATA}) — DLD open data (Dubai Pulse), snapshot to {D['date_max_clean']}.
* **License:** {D['license']} — "Dubai Real Estate Data (dubairealestatedata.com), based on Dubai Land Department open data."
  The data is not stored in this repo; download it from Kaggle into `data/`.
* Every row is a **registered sale**, not an asking price.

## What the data says
1. Sales exploded: **{F['tx_2025']:,}** home sales in 2025 (after cleaning) vs **{F['tx_2020']:,}** in 2020 ({F['tx_multiple_2025_vs_2020']}x).
2. Off-plan took over: **{F['offplan_share_2025_pct']}%** of 2025 home sales were off-plan, up from {F['offplan_share_2021_pct']}% in 2021.
3. Median price per m² went from AED {F['median_psm_2020']:,} (2020) to AED {F['median_psm_2025']:,} (2025): **+{F['median_psm_growth_2020_2025_pct']}%**.
4. Homes are shrinking: the median home sold in 2026 is **{F['median_size_2026ytd_sqm']} m²**, the smallest since 2010 (2021: {F['median_size_2021_sqm']} m²).
5. Cheaper areas caught up: **{F['fastest_area_2021_2025']}** rose **+{F['fastest_area_growth_pct']}%** per m² (2021→2025) while Marsa Dubai (Dubai Marina) rose {F['marina_growth_pct']}% (among {F['n_areas_compared_2021_2025']} areas with ≥300 sales in both years).
6. In 2025 off-plan sold at AED {F['offplan_psm_2025']:,}/m² vs AED {F['ready_psm_2025']:,} for ready homes (+{F['offplan_premium_2025_pct']}%) — this compares different homes in different areas, so it is not a like-for-like premium.
7. One area — {F['busiest_area_2025']} (marketed as Jumeirah Village Circle) — had **{F['busiest_area_2025_tx']:,}** sales in 2025, {F['busiest_area_2025_share_pct']}% of the whole market.

| | |
|---|---|
| ![](figures/01_volume_by_year.png) | ![](figures/02_psm_by_year.png) |
| ![](figures/03_area_growth.png) | ![](figures/04_size_by_year.png) |

## Problem set-up
* **Target:** log(price per m²) of a registered sale. Price = prediction × size, so % error on price = % error on price per m².
* **Rows:** residential units and villas, {D['rows_clean']:,} sales after cleaning ({D['date_min_clean']} → {D['date_max_clean']}).
* **Split (by date, no shuffling):** train {S['train']['from']} → {S['train']['to']} ({S['train']['n']:,}),
  validation {S['validation']['from']} → {S['validation']['to']} ({S['validation']['n']:,}),
  test {S['test']['from']} → {S['test']['to']} ({S['test']['n']:,}).
* **Main metric:** median absolute % error (MdAPE) on price, plus the share within ±10% / ±20% (how automated valuation models are judged).

### Cleaning (every step logged)
| Step | Why | Rows removed | Rows left |
|---|---|---:|---:|
{clean_rows}

![](figures/10_cleaning_funnel.png)

### How leakage is prevented
* All {S['n_features']} features are things known when the sale is registered. History features
  (past prices in the same building / project / area / segment / market) are computed from a **daily aggregate with a
  left-closed time window**: a sale on day *D* only sees sales registered on days **before** *D* — not even the same day.
* [`tests/test_no_leakage.py`](tests/test_no_leakage.py) proves it two ways: (1) recomputes features for random rows by brute force
  from earlier days only and checks they match; (2) scrambles every price from a cutoff day onward and checks that no feature of any
  sale on or before that day changes.
* Model choice, early stopping and Optuna tuning used the 2024 validation year only. The test period was scored once at the end.

## Results
### Validation year (2024) — used for all choices
| Model | MdAPE | within ±10% | within ±20% | R² (log price/m²) | |
|---|---:|---:|---:|---:|---|
{val_rows}

Optuna ({T['n_trials']} trials, LightGBM) moved validation MdAPE from {T['val_mdape_untuned']:.3f}% to {T['val_mdape_tuned']:.3f}% — a small gain.

### Test period ({S['test']['from']} → {S['test']['to']})
| Model | MdAPE | within ±10% | within ±20% | R² (log price/m²) | |
|---|---:|---:|---:|---:|---|
{test_rows}

Ridge's negative R² is real, not a typo: its errors grow sharply in 2026 (a linear model extrapolates badly once the market moves
outside the training range), so its squared error exceeds the spread of the prices themselves. The main metrics stay the median
error and the ±10% / ±20% shares.

| | |
|---|---|
| ![](figures/06_test_error_by_month.png) | ![](figures/07_pred_vs_actual.png) |

### Where it works and where it doesn't (test, frozen model)
| Segment | Sales | Model MdAPE | Building rule MdAPE |
|---|---:|---:|---:|
{seg_rows}

![](figures/09_error_by_segment.png)

### How sure is it?
This is regression, so instead of probability calibration the uncertainty is checked directly: an 80% interval built on
2024 residuals (split conformal, ±{I['half_width_log']} in log space ≈ {I['relative_band_pct'][0]}% / +{I['relative_band_pct'][1]}%)
covered **{I['test_coverage_pct']:.1f}%** of test sales (worst month {I['test_coverage_min_month_pct']:.1f}%).

### Why it predicts what it predicts (SHAP)
![](figures/08_shap_importance.png)

### Case studies (real test sales)
| Case | Date | Area | Home | Actual AED | Predicted AED | Error | Past sales in building | Top SHAP features |
|---|---|---|---|---:|---:|---:|---:|---|
{case_rows}

{misses_note}

## Limitations
* It predicts the **registered** price. It knows nothing about view, floor, finishing, or a motivated seller.
* It leans on recent sales nearby. New / unknown buildings are clearly harder.
* **It ages.** Frozen at the end of 2024, its error rose from {DR['mdape_first3_months_avg']:.2f}% (first 3 test months) to
  {DR['mdape_last3_months_avg']:.2f}% (last 3). It needs regular retraining; the walk-forward result above is the realistic number.
  The last test month ({DR['last_month']}) has only {DR['last_month_is_partial_days']} days of data.
* EDA medians mix different homes each year, so a rising median is not a like-for-like price index.
* Exact duplicate rows were dropped because identical bulk sales cannot be told apart from re-registrations.
* Ridge was run with light settings; Random Forest on a 300,000-row subsample for speed.
* PerpetualBooster was tried too, but its scores changed between two identical runs, so it was dropped to keep the project reproducible. It was never the best model.

## Project structure
```
src/features.py       cleaning + leakage-safe features
src/modeling.py       split, baselines, models, metrics, Optuna, conformal interval
src/style.py          one chart theme
src/make_dashboard.py offline Plotly dashboard (dashboard/dubai_price_dashboard.html)
src/make_readme.py    renders this README from results/metrics.json
notebooks/            main notebook (+ a self-contained Kaggle version)
app/streamlit_app.py  results explorer + what-if simulator
tests/                leakage tests
scripts/verify_numbers.py  checks that every number in a text file exists in metrics.json
results/              metrics.json, test predictions, what-if lookup
models/               LightGBM frozen at 2024 (evaluated) + production model (refit on all data, used by the app)
```

## How to run
```bash
pip install -r requirements.txt
# download dld_sales_transactions.parquet from the Kaggle dataset into data/
pytest -q tests/                      # leakage tests
jupyter nbconvert --to notebook --execute --inplace notebooks/dubai_price_model.ipynb
python src/make_readme.py             # rebuild this README from metrics.json
python scripts/verify_numbers.py README.md
streamlit run app/streamlit_app.py
```
Open `dashboard/dubai_price_dashboard.html` in any browser (works offline).

## License
Code: MIT (see `LICENSE`). Data: CC BY 4.0, Dubai Real Estate Data (dubairealestatedata.com) based on Dubai Land Department open data.

---
Built by **Hussain Ahmed** · [GitHub](https://github.com/HussainAhmed75) · [Kaggle](https://www.kaggle.com/hussianahmed)
"""
open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8").write(readme)
print("README.md written")
