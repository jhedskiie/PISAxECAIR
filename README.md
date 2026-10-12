# PISAxECAIR

## Set Up Instructions

1. Clone the repository

```bash
git clone https://github.com/jhedskiie/PISAxECAIR.git
cd PISAxECAIR
```

2. Set up the environment with uv

```bash
# Create virtual environment
uv venv .venv

# Activate the environment
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

3. Download the PISA data

Download the **SAS-format** school and student questionnaire files from the OECD PISA database for each cycle. Place them in `data/raw/<year>/`:

```
data/raw/2018/
├── CY07_MSU_SCH_QQQ.sas7bdat
└── CY07_MSU_STU_QQQ.sas7bdat

data/raw/2022/
├── CY08MSP_SCH_QQQ.SAS7BDAT
└── CY08MSP_STU_QQQ.SAS7BDAT
```

The exact file names don't matter. The pipeline finds the files automatically by pattern (`*SCH_QQQ*.sas7bdat` and `*STU_QQQ*.sas7bdat`, case-insensitive). Each year folder must contain exactly one school file and one student file. Other files in the folder, such as `.sas7bcat` format catalogs, are ignored.

> **Note:** The raw data files are large and are not tracked in Git. `data/` is listed in `.gitignore`.

---

## Dependencies

Declared in `pyproject.toml`:

- Preprocessing: pandas, numpy, scikit-learn, pyreadstat, rich
- Modelling: statsmodels, xgboost, optuna, shap, matplotlib

---

## Instructions

Run all commands from the project root (`PISAxECAIR/`).

### One-run pipeline (preprocessing + modelling)

`src/pipeline.py` runs the full analysis behind the September 2026 policy note, *School Conditions and Learning Outcomes in Southeast Asia*. It is a migration of Version 5 of PISAxECAIR-Research: `PisaxCair_Modeling_modularized.ipynb` (XGBoost, SHAP, counterfactuals) and `student_school_modeling_pisa_school_level_ols.ipynb` (regressions with the ESCS control).

1. **Preprocessing** (`preprocess.py`). Runs only if `data/processed/sch_combined_final_transformed.csv` is missing or lacks a requested cycle.
2. **School-mean ESCS** (`escs.py`). Reads student ESCS from the raw student files, averages it within each school, and caches it in `data/processed/sch_escs_combined.csv`. It is used only as a control in the regressions.
3. **Econometrics** (`econometrics.py`). Pooled OLS and country + cycle fixed effects for each domain: Table 8, Annex B and Annex C.
4. **Machine learning** (`ml_models.py`). XGBoost tuned with Optuna on a stratified 80/20 split, plus SHAP values. Covers pooled and per-cycle models (Tables 7, 9 and 10, Figures 5–7).
5. **Counterfactuals** (`counterfactuals.py`). Group actionable recourse for the Philippines toward Indonesia, the ASEAN average and Singapore, at λ = 1, 15 and 30, with bootstrap intervals: Tables 11–12 and Annex D.

```bash
# Everything, for all cycles in src/config.py
python src/pipeline.py

# Fast smoke test (5 Optuna trials per model, 200 per counterfactual search)
python src/pipeline.py --quick

# Selected stages / options
python src/pipeline.py --stages econometrics
python src/pipeline.py --stages ml counterfactuals
python src/pipeline.py --penalty-weights 15
python src/pipeline.py --preprocess always      # force re-running preprocessing
python src/pipeline.py --rebuild-escs           # re-read ESCS from the student files
```

A full run trains 9 tuned XGBoost models (100 Optuna trials × 3-fold CV each) and runs 9 counterfactual searches (5,000 trials each). Expect it to take tens of minutes. All settings (trials, search spaces, benchmarks, λ values, feasible ranges) are in the *Modelling* section of `src/config.py`.

### Preprocessing only

```bash
# Process all cycles listed in src/config.py
python src/preprocess.py

# Process specific cycles
python src/preprocess.py --years 2018 2022

# With PISA 2025 addition
python src/preprocess.py --years 2018 2022 2025
```

---

## Outputs

### `data/processed/`

| File | Description |
|------|-------------|
| `sch_sea<year>_imputed.csv` | Imputed school data for one cycle |
| `sch_combined_imputed.csv` | Imputed school data, all cycles stacked |
| `sch_combined_transformed.csv` | Derived variables, all cycles |
| `sch_perf_combined.csv` | School-level performance scores |
| `sch_combined_final_orig.csv` | Performance scores + **imputed raw questionnaire items** |
| `sch_combined_final_transformed.csv` | Performance scores + **derived variables**, ready for modeling |

Both final files have one row per school per cycle, with the same performance columns (`PVMATH_weighted`, `PVREAD_weighted`, `PVSCIE_weighted`, `ALL_DOMAINS_weighted`):
- **`final_orig`** keeps all original SC items, with their original codes and scales.
- **`final_transformed`** replaces them with the derived variables below. This is the modeling dataset.

Plus `sch_escs_combined.csv`: school-mean ESCS per school and cycle, written by the pipeline.

### `output/`

| File | Description |
|------|-------------|
| `country_means_summary.csv` | Country mean scores by cycle and domain, for checking against OECD's published results |

### `output/modeling/` (from `pipeline.py`)

| Path | Contents | Paper |
|------|----------|-------|
| `results_summary.md` | The headline tables of this run, in the paper's numbering | — |
| `run_info.json` | Run settings, timings and package versions | — |
| `econometrics/table_8_fixed_effects.csv` | FE coefficients with significance stars | Table 8 |
| `econometrics/table_B1_pooled_ols.csv` | Pooled OLS coefficients | Table B1 |
| `econometrics/table_C_fe_full_<DOMAIN>.csv` | Full FE results (SE, z, p, 95% CI) | Tables C1–C3 |
| `econometrics/coefficients_all.csv`, `model_fit.csv`, `vif.csv`, `diagnostics.csv`, `ols_summaries.txt` | All coefficients incl. fixed effects, fit statistics, VIFs, RESET / Breusch–Pagan / White / Jarque–Bera | — |
| `ml/table_9_xgb_performance.csv` | Test-set MAE / RMSE / R² by cycle and domain | Table 9 |
| `ml/table_10_shap_importance.csv` | Mean \|SHAP\| per predictor, cycle and domain | Table 10 |
| `ml/figures/shap_<cycle>_<DOMAIN>_{beeswarm,bar}.png` | SHAP plots (`pooled` = merged) | Figures 5–7 |
| `ml/figures/shap_pooled_<DOMAIN>_dependence_STU_TCH.png` | SHAP dependence plot for the student–teacher ratio | — |
| `ml/models/`, `ml/optuna_trials/`, `ml/best_params_and_metrics.json` | Fitted models (XGBoost JSON), Optuna logs, tuned parameters | Table 7 |
| `ml/permutation_importance.csv`, `ml/split_audit.csv` | Permutation importance, country × cycle split audit | — |
| `counterfactuals/lambda_<λ>/interventions.csv` | Recommended intervention package (means and medians) | Table 11, D1/D4/D7 |
| `counterfactuals/lambda_<λ>/predicted_outcomes.csv` | Baseline, benchmark, simulated score, gain, remaining gap, attainment | Table 12, D2/D5/D8 |
| `counterfactuals/lambda_<λ>/bootstrap.csv` | Bootstrap mean, SD of gain, 95% CI | D3/D6/D9 |
| `counterfactuals/lambda_<λ>/scenario_summary.csv` | Every scenario (actual, distribution transfer, recourse, target) with bootstrap CIs of the mean and of the paired gain | — |
| `counterfactuals/benchmark_targets.csv`, `feasible_ranges.csv`, `distribution_transfer_*.csv` | Benchmark scores, feasible intervention space, distribution-transfer scenario | — |

### Methodological notes for reproducing the paper

- **Row order.** Modelling data are sorted by cycle, then `CNTSCHID`, before splitting. This matches the Version 5 input and reproduces its train/test split exactly.
- **Standard errors.** OLS standard errors are clustered by `CNTSCHID` (`OLS_COV_TYPE = "cluster"`), which is what Version 5 ran and what reproduces the SEs in Annex C. The paper's table notes say "HC3". Set `OLS_COV_TYPE = "HC3"` to get those instead. Coefficients are unaffected.
- **Search spaces.** Pooled models use the search space in Table 7. Per-cycle models (2018/2022 rows of Tables 9–10) use the separate per-cycle space from Version 5 (`XGB_SPACE_CYCLE`).
- **Counterfactual sample.** Counterfactuals use the pooled MATH model and the 2022 test-set schools (`CF_DOMAIN`, `CF_YEAR`).

---

## Adding a New PISA Cycle

To add a new cycle (e.g., 2025):

1. Put the school and student SAS files in `data/raw/2025/`.
2. Add the cycle to `CYCLES` in `src/config.py`:

```python
CYCLES = {
    2018: {},
    2022: {},
    2025: {},
}
```

3. Run the pipeline:

```bash
python src/preprocess.py --years 2018 2022 2025
# or preprocessing + modelling in one go
python src/pipeline.py --years 2018 2022 2025 --preprocess always --rebuild-escs
```

The models pool every cycle you pass. Per-cycle models and the cycle strata adapt automatically. Counterfactuals use `CF_YEAR` in `src/config.py`; set it to `2025` to simulate on the newest cycle.

**If OECD renamed questionnaire items**, the pipeline stops and lists the missing item codes. Map each new code to the old one in that cycle's entry:

```python
2025: {"rename": {"SC016Q01JA": "SC016Q01TA"}},
```

If an item was removed entirely, update `data_schema.py` and any derived variable that uses it.

### Other cycle options

| Key | Purpose | Example |
|-----|---------|---------|
| `school` / `student` | Set exact file names instead of detecting them by pattern | `"school": "my_file.sas7bdat"` |
| `rename` | Map renamed item codes to the codes in `data_schema.py` | `{"NEW_CODE": "OLD_CODE"}` |
| `exclude_countries` | Manually drop countries from the performance data | `["VNM"]` |

---

## Configuration

All settings are in `src/config.py`:

| Setting | Description | Default |
|---------|-------------|---------|
| `SEA_COUNTRIES` | Countries included | 8 SEA countries |
| `CYCLES` | Cycles to process | 2018, 2022 |
| `DOMAINS` | Assessment domains | MATH, READ, SCIE |
| `RANDOM_STATE` | Random seed | 42 |
| `N_ESTIMATORS` | Trees per ExtraTrees model | 200 |
| `MAX_ITER` | Imputation iterations | 10 |
| `N_JOBS` | CPU cores (`-1` = all; results are unaffected) | -1 |

Modelling settings (used by `pipeline.py`):

| Setting | Description | Default |
|---------|-------------|---------|
| `FEATURES` | The 12 school-level predictors (paper Table 1) | — |
| `TARGETS` | Outcome column per domain | `PV<DOMAIN>_weighted` |
| `OLS_COV_TYPE` | OLS standard errors (`"cluster"` by school, or `"HC3"`, …) | `"cluster"` |
| `TEST_SIZE`, `CV_FOLDS`, `N_TRIALS` | Hold-out share, CV folds, Optuna trials per model | 0.20, 3, 100 |
| `XGB_SPACE_POOLED`, `XGB_SPACE_CYCLE` | Optuna search spaces (pooled = Table 7) | — |
| `SHAP_DEPENDENCE_FEATURES` | Features that get a SHAP dependence plot (pooled models) | `["STU_TCH"]` |
| `CF_DOMAIN`, `CF_SOURCE_COUNTRY`, `CF_YEAR` | Model, country and cycle for the simulations | MATH, PHL, 2022 |
| `CF_BENCHMARKS` | Target groups (ASEAN average excludes the source country) | ASEAN, IDN, SGP |
| `CF_PENALTY_WEIGHTS` | λ values | 1, 15, 30 |
| `CF_N_TRIALS` | Optuna trials per counterfactual search | 5000 |
| `CF_MUTABLE`, `CF_CONTINUOUS`, `CF_DOMAIN_RANGES`, `CF_QUANTILES` | Actionable features, non-integer features, feasible bounds | — |

---
