from __future__ import annotations

import argparse
import time
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pyreadstat
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import config as cfg
from data_schema import SCHEMA_GROUPS

try:
    from data_schema import COMPOSITION_GROUPS
except ImportError:
    COMPOSITION_GROUPS = {
        "SC016": {"cols": ["SC016Q01TA", "SC016Q02TA", "SC016Q03TA", "SC016Q04TA"], "sum": 100}
    }

from sklearn.exceptions import ConvergenceWarning

# IterativeImputer warns when it uses all MAX_ITER rounds; expected here.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

console = Console()

# Columns driven by the schema (single source of truth)
SCHEMA_COLS = list(dict.fromkeys(c for g in SCHEMA_GROUPS for c in g["cols"]))
COMP_COLS = list(dict.fromkeys(c for g in COMPOSITION_GROUPS.values() for c in g["cols"]))
NON_COMP_COLS = [c for c in SCHEMA_COLS if c not in COMP_COLS]
SCHOOL_KEEP_COLS = cfg.SCHOOL_ID_COLS + NON_COMP_COLS + COMP_COLS


# Progress helpers

_STEP = {"n": 0, "total": 0}


@contextmanager
def step(title: str):
    """Numbered section header with elapsed time."""
    _STEP["n"] += 1
    console.print(f"\n[bold cyan]Step {_STEP['n']}/{_STEP['total']} · {title}[/bold cyan]")
    t0 = time.perf_counter()
    yield
    console.print(f"[green]✓ done[/green] [dim]({time.perf_counter() - t0:.1f}s)[/dim]\n")


@contextmanager
def task(msg: str):
    """Spinner for a long-running sub-task."""
    t0 = time.perf_counter()
    with console.status(f"{msg}…"):
        yield
    console.print(f"  • {msg} [dim]({time.perf_counter() - t0:.1f}s)[/dim]")


def info(msg: str):
    console.print(f"  [dim]→[/dim] {msg}")


def warn(msg: str):
    console.print(f"  [yellow]⚠ {msg}[/yellow]")


# 1. Load

def find_file(year: int, kind: str) -> "Path":
    """Locate the school ('SCH') or student ('STU') SAS file for a cycle."""
    folder = cfg.DATA_RAW / str(year)
    explicit = cfg.CYCLES[year].get("school" if kind == "SCH" else "student")
    if explicit:
        path = folder / explicit
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    matches = [
        p for p in folder.glob("*")
        if p.suffix.lower() == ".sas7bdat" and f"{kind}_QQQ" in p.name.upper()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *{kind}_QQQ*.sas7bdat in {folder}, found {len(matches)}: "
            f"{[m.name for m in matches]}. Set the file name in config.CYCLES[{year}]."
        )
    return matches[0]


def read_sas_columns(path, wanted: list[str] | None = None, prefixes: tuple = ()) -> pd.DataFrame:
    """Read only the needed columns of a SAS file (much faster on student files)."""
    _, meta = pyreadstat.read_sas7bdat(str(path), metadataonly=True)
    available = meta.column_names
    cols = [c for c in available if (wanted and c in wanted) or c.startswith(prefixes)]
    df, _ = pyreadstat.read_sas7bdat(str(path), usecols=cols)
    return df


def is_pv_col(col: str) -> bool:
    return col.startswith("PV") and col.endswith(tuple(cfg.DOMAINS))


def load_cycle(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the school and student data for one cycle, filtered to SEA countries."""
    rename = cfg.CYCLES[year].get("rename", {})
    rename_back = {v: k for k, v in rename.items()}  # read the cycle's own codes

    sch_path, stu_path = find_file(year, "SCH"), find_file(year, "STU")

    with task(f"{year} school file  [dim]{sch_path.name}[/dim]"):
        wanted = [rename_back.get(c, c) for c in SCHOOL_KEEP_COLS]
        sch = read_sas_columns(sch_path, wanted).rename(columns=rename)

    with task(f"{year} student file [dim]{stu_path.name}[/dim]"):
        stu = read_sas_columns(stu_path, cfg.STUDENT_ID_COLS, prefixes=("PV",))
        stu = stu[[c for c in stu.columns if c in cfg.STUDENT_ID_COLS or is_pv_col(c)]]

    sch = sch[sch[cfg.COUNTRY_COL].isin(cfg.SEA_COUNTRIES)].reset_index(drop=True)
    stu = stu[stu[cfg.COUNTRY_COL].isin(cfg.SEA_COUNTRIES)].reset_index(drop=True)
    return sch, stu


# 2. Clean

def prepare_school(sch: pd.DataFrame, year: int) -> pd.DataFrame:
    """Keep IDs + schema items in a fixed order; fail loudly if items are missing."""
    missing = [c for c in SCHOOL_KEEP_COLS if c not in sch.columns]
    if missing:
        raise KeyError(
            f"{year}: missing schema items {missing}. Map renamed items in "
            f"config.CYCLES[{year}]['rename'] or update data_schema.py."
        )
    df = sch[SCHOOL_KEEP_COLS].copy()
    df[SCHEMA_COLS] = df[SCHEMA_COLS].apply(pd.to_numeric, errors="coerce")
    return df

# 3. Imputation

def make_imputer() -> IterativeImputer:
    return IterativeImputer(
        estimator=ExtraTreesRegressor(
            n_estimators=cfg.N_ESTIMATORS, random_state=cfg.RANDOM_STATE, n_jobs=cfg.N_JOBS
        ),
        max_iter=cfg.MAX_ITER,
        random_state=cfg.RANDOM_STATE,
    )


# Compositional helpers (ILR)

def closure(X, total=100.0):
    X = np.asarray(X, dtype=float)
    return X / X.sum(axis=1, keepdims=True) * total


def make_strictly_positive(X, eps=1e-6):
    X = np.asarray(X, dtype=float)
    return np.where(X <= 0, eps, X)


def helmert_submatrix(D):
    H = np.zeros((D - 1, D))
    for i in range(1, D):
        H[i - 1, :i] = 1 / np.sqrt(i * (i + 1))
        H[i - 1, i] = -i / np.sqrt(i * (i + 1))
    return H


def ilr_transform(X):
    logX = np.log(X)
    clr = logX - logX.mean(axis=1, keepdims=True)
    return clr @ helmert_submatrix(X.shape[1]).T


def ilr_inverse(Z):
    return np.exp(Z @ helmert_submatrix(Z.shape[1] + 1))


#  Imputation steps

def fix_composition_simple(df: pd.DataFrame, cols: list[str], total: float) -> pd.DataFrame:
    """
    Case A: exactly 1 part missing  → fill with total − sum(others).
    Case B: no part missing         → rescale so parts sum to total.
    """
    comp = df[cols].copy()
    miss_k = comp.isna().sum(axis=1)

    mask_one = miss_k == 1
    fill_val = total - comp.sum(axis=1, skipna=True)
    for col in cols:
        m = mask_one & comp[col].isna()
        comp.loc[m, col] = fill_val[m]

    row_sum = comp.sum(axis=1)
    mask_full = (miss_k == 0) & (row_sum > 0)
    comp.loc[mask_full, cols] = comp.loc[mask_full, cols].div(row_sum[mask_full], axis=0) * total

    df[cols] = comp.clip(0, total)
    info(f"{', '.join(cols[:1])}…: filled {int(mask_one.sum())} rows with 1 missing, "
         f"rescaled {int(mask_full.sum())} complete rows")
    return df


def impute_non_compositional(df: pd.DataFrame) -> pd.DataFrame:
    """IterativeImputer (ExtraTrees) on ordinal/binary/count items."""
    n_missing = int(df[NON_COMP_COLS].isna().sum().sum())
    with task(f"Iterative imputation of {len(NON_COMP_COLS)} items ({n_missing:,} missing cells)"):
        imputed = make_imputer().fit_transform(df[NON_COMP_COLS])
    df[NON_COMP_COLS] = pd.DataFrame(imputed, columns=NON_COMP_COLS, index=df.index)
    return df


def apply_schema_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Round and clip imputed values to each item's valid range."""
    for g in SCHEMA_GROUPS:
        cols = [c for c in g["cols"] if c in df.columns]
        post = g.get("post", [])
        if "round" in post:
            df[cols] = df[cols].round()
        if "clip" in post:
            df[cols] = df[cols].clip(lower=g.get("min"), upper=g.get("max"))
    return df


def impute_composition_ilr(df: pd.DataFrame, cols: list[str], total: float) -> pd.DataFrame:
    """Impute rows with ≥2 missing parts in ILR space, using non-compositional items."""
    comp = df[cols].copy()
    miss_k = comp.isna().sum(axis=1)
    mask_train, mask_hard = miss_k == 0, miss_k >= 2

    if not mask_hard.any():
        info("No rows need ILR imputation")
        return df

    Z_train = ilr_transform(closure(make_strictly_positive(comp.loc[mask_train].to_numpy()), total))
    z_cols = [f"ILR_{i}" for i in range(Z_train.shape[1])]

    train = df.loc[mask_train, NON_COMP_COLS].copy()
    train[z_cols] = Z_train
    test = df.loc[mask_hard, NON_COMP_COLS].copy()
    test[z_cols] = np.nan
    combo = pd.concat([train, test], ignore_index=True)

    with task(f"ILR imputation for {int(mask_hard.sum())} rows"):
        combo_imp = pd.DataFrame(make_imputer().fit_transform(combo), columns=combo.columns)

    Z_imp = combo_imp.loc[len(train):, z_cols].to_numpy()
    df.loc[mask_hard, cols] = closure(make_strictly_positive(ilr_inverse(Z_imp)), total)
    return df


def validate_imputation(df: pd.DataFrame, year: int):
    n_nan = int(df[SCHEMA_COLS].isna().sum().sum())
    if n_nan:
        raise ValueError(f"{year}: {n_nan} NaNs remain after imputation")
    for name, g in COMPOSITION_GROUPS.items():
        off = int((df[g["cols"]].sum(axis=1).round(2) != g["sum"]).sum())
        if off:
            warn(f"{name}: {off} rows do not sum to {g['sum']} (e.g. all parts reported as 0)")
    info(f"No missing values remain · shape {df.shape}")


def impute_cycle(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Full imputation for one cycle, in the original notebook order."""
    df = df.copy()
    for g in COMPOSITION_GROUPS.values():
        df = fix_composition_simple(df, g["cols"], g["sum"])
    df = impute_non_compositional(df)
    df = apply_schema_groups(df)
    for g in COMPOSITION_GROUPS.values():
        df = impute_composition_ilr(df, g["cols"], g["sum"])
    validate_imputation(df, year)
    return df


# 4. Transform (based on ECAIR remarks)

DIGI_CAP_COLS = ["SC155Q06HA", "SC155Q07HA", "SC155Q08HA", "SC155Q09HA", "SC155Q10HA", "SC155Q11HA"]
EXTRA_CURRICULAR_COLS = ["SC053Q01TA", "SC053Q02TA", "SC053Q03TA", "SC053Q04TA", "SC053Q09TA", "SC053Q10TA"]
PARENT_ENG_COLS = ["SC064Q01TA", "SC064Q02TA", "SC064Q03TA", "SC064Q04NA"]

# Source items dropped after building the indices. Retained items:
# SC013Q01TA (public/private), SC037Q02TA (external evaluation),
# SC061Q01TA (truancy), SC025Q01NA (professional development).
TRANSFORM_DROP_COLS = [
    "SC001Q01TA", "SC016Q01TA", "SC016Q02TA", "SC016Q03TA", "SC016Q04TA",
    "SC017Q01NA", "SC017Q02NA", "SC017Q03NA", "SC017Q04NA", "SC017Q05NA",
    "SC017Q06NA", "SC017Q07NA", "SC017Q08NA",
    *DIGI_CAP_COLS,
    "SC011Q01TA", "SC012Q01TA", "SC012Q02TA", "SC012Q03TA", "SC012Q04TA", "SC012Q05TA", "SC012Q06TA",
    "SC042Q01TA", "SC042Q02TA",
    "SC037Q01TA", "SC037Q03TA", "SC037Q04TA", "SC037Q05NA", "SC037Q06NA",
    "SC037Q07TA", "SC037Q08TA", "SC037Q09TA",
    "SC061Q02TA", "SC061Q03TA", "SC061Q04TA", "SC061Q05TA", "SC061Q06TA",
    "SC061Q07TA", "SC061Q08TA", "SC061Q09TA", "SC061Q10TA", "SC061Q11HA",
    "SC002Q01TA", "SC002Q02TA", "SC018Q01TA01", "SC018Q01TA02", "SC018Q02TA01", "SC018Q02TA02",
    "SC004Q01TA", "SC004Q02TA", "SC004Q03TA", "SC004Q05NA", "SC004Q06NA", "SC004Q07NA",
    "SC003Q01TA",
    *EXTRA_CURRICULAR_COLS,
    *PARENT_ENG_COLS,
    "TOT_STU", "TOT_TCH",
]


def safe_ratio(num: pd.Series, den: pd.Series, name: str) -> pd.Series:
    """num / den, with 0 where den is 0 (avoids inf in model inputs)."""
    n_zero = int((den == 0).sum())
    if n_zero:
        info(f"{name}: {n_zero} rows with zero denominator set to 0")
    return (num / den.replace(0, np.nan)).fillna(0).round(4)


def transform_school(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # External (non-government) funding share: student fees + donations + others
    df["EXT_FUND"] = df["SC016Q02TA"] + df["SC016Q03TA"] + df["SC016Q04TA"]

    # Proxies: inadequate/poor educational material and physical infrastructure
    df["EDU_MAT"] = df["SC017Q06NA"]
    df["EDU_INFRA"] = df["SC017Q08NA"]

    # Digital capacity index: mean of indicators
    df["DIGI_CAP"] = df[DIGI_CAP_COLS].mean(axis=1).round(0)

    # Ratios (full-time + half of part-time teachers)
    df["TOT_STU"] = df["SC002Q01TA"] + df["SC002Q02TA"]
    df["TOT_TCH"] = df["SC018Q01TA01"] + df["SC018Q01TA02"] / 2
    df["STU_TCH"] = safe_ratio(df["TOT_STU"], df["TOT_TCH"], "STU_TCH")
    df["STU_COMP"] = safe_ratio(df["SC004Q01TA"], df["SC004Q03TA"], "STU_COMP")
    df["TCH_COMP"] = safe_ratio(df["TOT_TCH"], df["SC004Q07NA"], "TCH_COMP")

    # Extra-curricular offerings: count offered (1 = yes, 2 = no → 0)
    df["X_ACTIV"] = df[EXTRA_CURRICULAR_COLS].replace(2, 0).sum(axis=1)

    # Parent engagement index: mean of indicators
    df["PRNT_ENG"] = df[PARENT_ENG_COLS].mean(axis=1)

    df = df.drop(columns=TRANSFORM_DROP_COLS)
    info(f"Transformed shape {df.shape}")
    return df


# 5. School performance from student plausible values

def get_pv_cols(df: pd.DataFrame, domain: str) -> list[str]:
    pv = [c for c in df.columns if c.startswith("PV") and c.endswith(domain)]
    return sorted(pv, key=lambda x: int(x[2:-len(domain)]))


def drop_countries_without_pvs(stu: pd.DataFrame, year: int) -> pd.DataFrame:
    """Drop countries with no plausible values in any domain (e.g. VNM 2018)."""
    exclude = set(cfg.CYCLES[year].get("exclude_countries", []))
    for cnt, grp in stu.groupby(cfg.COUNTRY_COL):
        empty = [d for d in cfg.DOMAINS if grp[get_pv_cols(grp, d)].isna().all().all()]
        if empty:
            exclude.add(cnt)
            warn(f"{year} {cnt}: no plausible values for {', '.join(empty)} → dropped")
    return stu[~stu[cfg.COUNTRY_COL].isin(exclude)].reset_index(drop=True)


def school_performance(stu: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted school mean per PV, then average across PVs (PISA approach).
    One row per (CNT, CNTSCHID): PV<DOMAIN>_weighted + ALL_DOMAINS_weighted.
    """
    keys = [cfg.COUNTRY_COL, cfg.SCHOOL_COL]
    w = pd.to_numeric(stu[cfg.WEIGHT_COL], errors="coerce")
    out = stu[keys].drop_duplicates().reset_index(drop=True)

    for domain in cfg.DOMAINS:
        pvs = get_pv_cols(stu, domain)
        X = stu[pvs].apply(pd.to_numeric, errors="coerce")
        valid = X.notna() & w.notna().to_numpy()[:, None]

        xw = X.where(valid).mul(w, axis=0).add_suffix("_xw")
        ww = valid.mul(w.fillna(0), axis=0).add_suffix("_w")
        sums = pd.concat([stu[keys], xw, ww], axis=1).groupby(keys).sum(min_count=1)

        school_pv_means = sums[xw.columns].to_numpy() / sums[ww.columns].to_numpy()
        dom = pd.DataFrame({f"PV{domain}_weighted": np.nanmean(school_pv_means, axis=1)},
                           index=sums.index).reset_index()
        out = out.merge(dom, on=keys, how="left")

    out["ALL_DOMAINS_weighted"] = out[[f"PV{d}_weighted" for d in cfg.DOMAINS]].mean(axis=1)
    return out


def country_means(stu: pd.DataFrame) -> pd.DataFrame:
    """Country mean per domain: weighted mean per PV, averaged across PVs (no SEs)."""
    w = pd.to_numeric(stu[cfg.WEIGHT_COL], errors="coerce")
    rows = []
    for cnt, grp in stu.groupby(cfg.COUNTRY_COL):
        row = {cfg.COUNTRY_COL: cnt}
        gw = w.loc[grp.index]
        for domain in cfg.DOMAINS:
            pv_means = []
            for pv in get_pv_cols(grp, domain):
                x = pd.to_numeric(grp[pv], errors="coerce")
                m = x.notna() & gw.notna()
                pv_means.append((x[m] * gw[m]).sum() / gw[m].sum())
            row[f"{domain}_mean"] = np.mean(pv_means)
        rows.append(row)
    return pd.DataFrame(rows)


# Output helpers

def save(df: pd.DataFrame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    info(f"saved [bold]{path.relative_to(cfg.ROOT)}[/bold]  {df.shape}")


def print_country_means(df: pd.DataFrame):
    table = Table(title="Country means (compare with OECD PISA results)", header_style="bold")
    for col in df.columns:
        table.add_column(col, justify="right" if col.endswith("_mean") else "left")
    for _, r in df.iterrows():
        table.add_row(*[f"{v:.0f}" if isinstance(v, float) else str(v) for v in r])
    console.print(table)


# Main

def run(years: list[int], loader=load_cycle):
    unknown = [y for y in years if y not in cfg.CYCLES]
    if unknown:
        raise SystemExit(f"Years {unknown} are not in config.CYCLES")

    _STEP["total"] = 6
    console.print(f"[bold]PISA preprocessing · cycles {', '.join(map(str, years))}")
    info(f"{len(SCHEMA_COLS)} schema items · {len(COMP_COLS)} compositional · "
         f"{len(cfg.SEA_COUNTRIES)} countries\n")
    t_start = time.perf_counter()

    # 1. Load & clean
    school, student = {}, {}
    with step("Load and clean raw data"):
        for y in years:
            sch, stu = loader(y)
            school[y] = prepare_school(sch, y)
            student[y] = stu
            info(f"{y}: {len(school[y]):,} schools · {len(stu):,} students")

    # 2. Impute each cycle
    imputed = {}
    with step("Impute school data (per cycle)"):
        for y in years:
            console.print(f"[bold]{y}[/bold]")
            imputed[y] = impute_cycle(school[y], y).assign(**{cfg.YEAR_COL: y})
            save(imputed[y], cfg.DATA_PROCESSED / f"sch_sea{y}_imputed.csv")
        sch_combined = pd.concat(imputed.values(), ignore_index=True)
        save(sch_combined, cfg.DATA_PROCESSED / "sch_combined_imputed.csv")

    # 3. Transform
    with step("Transform school variables"):
        sch_transformed = transform_school(sch_combined)
        save(sch_transformed, cfg.DATA_PROCESSED / "sch_combined_transformed.csv")

    # 4. School performance
    perf, means = [], []
    with step("Compute school performance from plausible values"):
        for y in years:
            stu = drop_countries_without_pvs(student[y], y)
            with task(f"{y}: weighted school means"):
                perf.append(school_performance(stu).assign(**{cfg.YEAR_COL: y}))
            means.append(country_means(stu).assign(**{cfg.YEAR_COL: y}))
        sch_perf = pd.concat(perf, ignore_index=True)
        save(sch_perf, cfg.DATA_PROCESSED / "sch_perf_combined.csv")

    # 5. Country means for manual validation
    with step("Country means summary"):
        summary = pd.concat(means, ignore_index=True)
        summary = summary[[cfg.YEAR_COL, cfg.COUNTRY_COL] + [f"{d}_mean" for d in cfg.DOMAINS]]
        print_country_means(summary)
        save(summary.round(2), cfg.OUTPUT_DIR / "country_means_summary.csv")

    # 6. Merge performance with school data
    with step("Merge performance with school data"):
        keys = [cfg.COUNTRY_COL, cfg.SCHOOL_COL, cfg.YEAR_COL]
        final_orig = sch_perf.merge(sch_combined, on=keys, how="left")
        final_trans = sch_perf.merge(sch_transformed, on=keys, how="left")
        n_unmatched = int(final_orig[cfg.SCHOOL_ID_COLS[-1]].isna().sum())
        if n_unmatched:
            warn(f"{n_unmatched} schools have scores but no school questionnaire data")
        save(final_orig, cfg.DATA_PROCESSED / "sch_combined_final_orig.csv")
        save(final_trans, cfg.DATA_PROCESSED / "sch_combined_final_transformed.csv")

    console.print(f"[bold green]Finished in {time.perf_counter() - t_start:.0f}s[/bold green]")


def main():
    parser = argparse.ArgumentParser(description="PISA school preprocessing pipeline")
    parser.add_argument("--years", type=int, nargs="+", default=list(cfg.CYCLES),
                        help="cycles to process (default: all in config.CYCLES)")
    run(parser.parse_args().years)


if __name__ == "__main__":
    main()