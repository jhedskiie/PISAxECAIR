"""
School-mean ESCS, the socioeconomic control used by the econometric models.

Version 5 read student ESCS from the PISA student files, averaged it within each
school and cycle (unweighted, missing values skipped) and joined it to the
school-level data. ESCS is a control only and is never given to XGBoost
(paper section 5.2).
"""
from __future__ import annotations

import pandas as pd

import config as cfg
import preprocess as pp
from reporting import info, save_csv, task, warn

KEYS = [cfg.COUNTRY_COL, cfg.SCHOOL_COL, cfg.YEAR_COL]


def school_escs_for_cycle(year: int) -> pd.DataFrame:
    path = pp.find_file(year, "STU")
    with task(f"{year} student ESCS [dim]{path.name}[/dim]"):
        stu = pp.read_sas_columns(path, [cfg.COUNTRY_COL, cfg.SCHOOL_COL, cfg.ESCS_RAW_COL])
    if cfg.ESCS_RAW_COL not in stu.columns:
        raise KeyError(f"{year}: column {cfg.ESCS_RAW_COL!r} not found in {path.name}")
    return school_escs_from_students(stu, year)


def school_escs_from_students(stu: pd.DataFrame, year: int) -> pd.DataFrame:
    """Unweighted mean ESCS per school (CNT, CNTSCHID) for one cycle."""
    stu = stu[stu[cfg.COUNTRY_COL].isin(cfg.SEA_COUNTRIES)]
    escs = pd.to_numeric(stu[cfg.ESCS_RAW_COL], errors="coerce")
    out = (
        stu.assign(**{cfg.ESCS_RAW_COL: escs})
        .groupby([cfg.COUNTRY_COL, cfg.SCHOOL_COL], as_index=False)[cfg.ESCS_RAW_COL]
        .agg(["mean", "count"])
        .rename(columns={"mean": cfg.ESCS_COL, "count": "n_students_escs"})
    )
    out[cfg.YEAR_COL] = year
    return out[KEYS + [cfg.ESCS_COL, "n_students_escs"]]


def build_school_escs(years: list[int], rebuild: bool = False) -> pd.DataFrame:
    """Return school-mean ESCS for `years`, reading the cache when it covers them."""
    if cfg.ESCS_FILE.exists() and not rebuild:
        cached = pd.read_csv(cfg.ESCS_FILE)
        if set(years) <= set(cached[cfg.YEAR_COL].unique()):
            info(f"Using cached {cfg.ESCS_FILE.name} (pass --rebuild-escs to refresh)")
            return cached[cached[cfg.YEAR_COL].isin(years)].reset_index(drop=True)

    frames = [school_escs_for_cycle(y) for y in years]
    escs = pd.concat(frames, ignore_index=True)
    n_missing = int(escs[cfg.ESCS_COL].isna().sum())
    if n_missing:
        warn(f"{n_missing} schools have no student ESCS; they are dropped from the regressions")
    save_csv(escs, cfg.ESCS_FILE)
    return escs
