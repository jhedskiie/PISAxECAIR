from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "Output"

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
N_JOBS = -1