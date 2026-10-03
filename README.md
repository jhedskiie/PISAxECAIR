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

- pandas
- numpy
- scikit-learn
- pyreadstat
- rich

---

## Instructions

Run all commands from the project root (`PISAxECAIR/`).

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

### `Output/`

| File | Description |
|------|-------------|
| `country_means_summary.csv` | Country mean scores by cycle and domain, for checking against OECD's published results |

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
```

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

---
