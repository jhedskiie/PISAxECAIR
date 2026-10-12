from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "output"

# Optional keys per cycle:
#   "school" / "student": exact file name inside data/raw/<year>/
#                         (omit to auto-detect by pattern)
#   "rename":             {new_code: old_code} for renamed items
#   "exclude_countries":  CNT codes to drop from the performance data
CYCLES = {
    2018: {},
    2022: {},
    # 2025: {},
}

# Sample
SEA_COUNTRIES = ["BRN", "IDN", "KHM", "MYS", "PHL", "SGP", "THA", "VNM"]
# No Myanmar or Timor-Leste in PISA.

# Identifiers
COUNTRY_COL = "CNT"
SCHOOL_COL = "CNTSCHID"
YEAR_COL = "YEAR"
WEIGHT_COL = "W_FSTUWT"
SCHOOL_ID_COLS = [COUNTRY_COL, SCHOOL_COL, "STRATUM"]
STUDENT_ID_COLS = [COUNTRY_COL, SCHOOL_COL, WEIGHT_COL]

DOMAINS = ["MATH", "READ", "SCIE"]

# Imputation
RANDOM_STATE = 42
N_ESTIMATORS = 200
MAX_ITER = 10
N_JOBS = 2


# ---------------------------------------------------------------------------
# Modelling (Version 5 of PISAxECAIR-Research, as reported in the
# September 2026 policy note). Used by src/pipeline.py and the modules it calls.
# ---------------------------------------------------------------------------

MODEL_INPUT = DATA_PROCESSED / "sch_combined_final_transformed.csv"  # from preprocess.py
ESCS_FILE = DATA_PROCESSED / "sch_escs_combined.csv"                 # school-mean ESCS
MODEL_OUTPUT_DIR = OUTPUT_DIR / "modeling"

# Outcome per domain (school-level weighted PV mean from preprocess.py)
TARGETS = {
    "READ": "PVREAD_weighted",
    "SCIE": "PVSCIE_weighted",
    "MATH": "PVMATH_weighted",
}

# The 12 school-level predictors (paper Table 1). Order is kept from the
# Version 5 notebooks because XGBoost column sampling depends on it.
FEATURES = [
    "SC013Q01TA",  # Public (1) / private (2)
    "SC037Q02TA",  # External evaluation
    "SC061Q01TA",  # Student truancy
    "SC025Q01NA",  # Teacher professional development (%)
    "EDU_MAT",
    "EDU_INFRA",
    "DIGI_CAP",
    "STU_TCH",
    "STU_COMP",
    "TCH_COMP",
    "X_ACTIV",
    "PRNT_ENG",
]

FEATURE_LABELS = {
    "SC061Q01TA": "Student Truancy",
    "X_ACTIV": "Extracurricular Activities",
    "EDU_MAT": "Educational Materials",
    "STU_TCH": "Student–Teacher Ratio",
    "DIGI_CAP": "Digital Capacity",
    "PRNT_ENG": "Parental Engagement",
    "SC025Q01NA": "Teacher Prof. Dev.",
    "STU_COMP": "Student–Computer Ratio",
    "TCH_COMP": "Teacher–Computer Ratio",
    "EDU_INFRA": "Infrastructure Constraint",
    "SC013Q01TA": "Private School",
    "SC037Q02TA": "External Evaluation",
    "mean_ESCS": "Economic, Social and Cultural Status",
}

# Econometrics (paper section 5.4.1, Table 8, Annex B and C)
ESCS_RAW_COL = "ESCS"          # student-level column in the STU file
ESCS_COL = "mean_ESCS"         # unweighted school mean, used as a control only
# "cluster" clusters by CNTSCHID, which is what Version 5 ran and reproduces the
# standard errors printed in Annex C. Any statsmodels cov_type ("HC3", ...) also works.
OLS_COV_TYPE = "cluster"

# XGBoost (paper section 5.4.2, Tables 7, 9 and 10)
TEST_SIZE = 0.20               # untouched test set, stratified by country × cycle
CV_FOLDS = 3
N_TRIALS = 100                 # Optuna trials per model
XGB_N_JOBS = 2
MAX_SHAP_ROWS = 1000

# Search space for the pooled (2018 + 2022) models, paper Table 7.
# Entries: (type, low, high, options). Order matters for TPE reproducibility.
XGB_SPACE_POOLED = {
    "n_estimators": ("int", 300, 1200, {}),
    "learning_rate": ("float", 0.005, 0.05, {"log": True}),
    "max_depth": ("int", 3, 10, {}),
    "min_child_weight": ("float", 1, 20, {}),
    "subsample": ("float", 0.6, 1.0, {}),
    "colsample_bytree": ("float", 0.6, 1.0, {}),
    "reg_lambda": ("float", 1e-2, 20.0, {"log": True}),
    "reg_alpha": ("float", 1e-4, 10.0, {"log": True}),
}
XGB_FIXED_POOLED = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
}

# Search space Version 5 used for the single-cycle models (2018 and 2022 rows of
# Tables 9 and 10). It differs from Table 7; kept as-is to reproduce those rows.
XGB_SPACE_CYCLE = {
    "n_estimators": ("int", 600, 1200, {"step": 50}),
    "learning_rate": ("float", 0.005, 0.03, {"log": True}),
    "max_depth": ("int", 4, 12, {}),
    "subsample": ("float", 0.7, 1.0, {}),
    "colsample_bytree": ("float", 0.7, 1.0, {}),
    "reg_lambda": ("float", 1.0, 20.0, {"log": True}),
    "min_child_weight": ("float", 1.0, 20.0, {"log": True}),
    "gamma": ("float", 0.0, 5.0, {}),
    "reg_alpha": ("float", 0.0, 10.0, {}),
}
XGB_FIXED_CYCLE = {
    "objective": "reg:squarederror",
}

# SHAP dependence plots drawn for the pooled models
SHAP_DEPENDENCE_FEATURES = ["STU_TCH"]

# Group counterfactual / actionable recourse (paper section 5.4.3, 6.3, Annex D)
CF_DOMAIN = "MATH"             # pooled model used for the simulations
CF_SOURCE_COUNTRY = "PHL"
CF_YEAR = 2022                 # test-set schools from this cycle
CF_BENCHMARKS = {
    # name: (countries, exclude source country from the group)
    "ASEAN average": (["BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "VNM", "TLS"], True),
    "Indonesia (IDN)": (["IDN"], False),
    "Singapore (SGP)": (["SGP"], False),
}
CF_PENALTY_WEIGHTS = [1.0, 15.0, 30.0]   # λ; Annex D tables D1–D9
CF_N_TRIALS = 5000             # Optuna TPE trials per benchmark and λ
CF_MAX_FEATURES_CHANGED = 12
CF_BOOTSTRAP = 1000

CF_MUTABLE = [
    "SC061Q01TA", "EDU_MAT", "EDU_INFRA", "DIGI_CAP", "STU_TCH", "STU_COMP",
    "TCH_COMP", "X_ACTIV", "PRNT_ENG", "SC025Q01NA", "SC037Q02TA",
]
# Only these mutable features may take non-integer values.
CF_CONTINUOUS = ["SC025Q01NA", "PRNT_ENG", "TCH_COMP", "STU_COMP", "STU_TCH"]
# Feasible region: training-set 1st–99th percentile, intersected with these bounds.
CF_QUANTILES = (0.01, 0.99)
CF_DOMAIN_RANGES = {
    "EDU_MAT": (1, 4),
    "EDU_INFRA": (1, 4),
    "DIGI_CAP": (1, 4),
    "STU_TCH": (0, 200),
    "STU_COMP": (0, 400),
    "TCH_COMP": (0, 200),
    "X_ACTIV": (0, 6),
    "PRNT_ENG": (0, 100),
}