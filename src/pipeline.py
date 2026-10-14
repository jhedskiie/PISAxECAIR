"""
One-run PISA × ECAIR pipeline: preprocessing → ESCS → econometrics → XGBoost/SHAP
→ group counterfactuals. Reproduces the analysis in the September 2026 policy note
("School Conditions and Learning Outcomes in Southeast Asia"), i.e. Version 5 of
PISAxECAIR-Research.

Run from the project root:

    python src/pipeline.py                      # everything, all cycles in config.CYCLES
    python src/pipeline.py --quick              # fast smoke test (few Optuna trials)
    python src/pipeline.py --stages econometrics
    python src/pipeline.py --preprocess always  # force re-running preprocess.py
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import platform
import time
from datetime import datetime

import pandas as pd

import config as cfg
from reporting import console, info, save_json, set_total_steps, step, warn

STAGES = ["econometrics", "ml", "counterfactuals"]
QUICK = {"n_trials": 5, "cf_trials": 200}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PISA × ECAIR one-run pipeline (Version 5 analysis)")
    p.add_argument("--years", type=int, nargs="+", default=list(cfg.CYCLES),
                   help="cycles to use (default: all in config.CYCLES)")
    p.add_argument("--preprocess", choices=["auto", "always", "skip"], default="auto",
                   help="auto: run preprocess.py only if its output is missing or lacks a cycle")
    p.add_argument("--rebuild-escs", action="store_true", help="re-read ESCS from the student files")
    p.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES, help="modelling stages to run")
    p.add_argument("--n-trials", type=int, default=cfg.N_TRIALS, help="Optuna trials per XGBoost model")
    p.add_argument("--cf-trials", type=int, default=cfg.CF_N_TRIALS,
                   help="Optuna trials per counterfactual search")
    p.add_argument("--penalty-weights", type=float, nargs="+", default=cfg.CF_PENALTY_WEIGHTS,
                   help="λ values for the group recourse")
    p.add_argument("--quick", action="store_true",
                   help=f"smoke test: {QUICK['n_trials']} XGBoost trials, {QUICK['cf_trials']} CF trials")
    args = p.parse_args(argv)
    if args.quick:
        args.n_trials, args.cf_trials = QUICK["n_trials"], QUICK["cf_trials"]
    return args


def needs_preprocessing(years: list[int]) -> bool:
    if not cfg.MODEL_INPUT.exists():
        return True
    have = set(pd.read_csv(cfg.MODEL_INPUT, usecols=[cfg.YEAR_COL])[cfg.YEAR_COL].unique())
    return not set(years) <= have


def load_model_data(years: list[int], escs: pd.DataFrame | None) -> pd.DataFrame:
    df = pd.read_csv(cfg.MODEL_INPUT)
    # Row order feeds the stratified train/test split, so fix it: cycle, then school ID.
    # This is the order of the Version 5 input and reproduces its exact split.
    df = (
        df[df[cfg.YEAR_COL].isin(years)]
        .sort_values([cfg.YEAR_COL, cfg.SCHOOL_COL], kind="mergesort")
        .reset_index(drop=True)
    )
    keys = [cfg.COUNTRY_COL, cfg.SCHOOL_COL, cfg.YEAR_COL]
    if df.duplicated(keys).any():
        raise ValueError(f"{cfg.MODEL_INPUT.name} has duplicate {keys} rows")
    if escs is not None:
        df = df.merge(escs[keys + [cfg.ESCS_COL]], on=keys, how="left", validate="one_to_one")
        n_missing = int(df[cfg.ESCS_COL].isna().sum())
        if n_missing:
            warn(f"{n_missing} schools without mean ESCS (dropped from the regressions only)")
    missing = [c for c in cfg.FEATURES + list(cfg.TARGETS.values()) if c not in df.columns]
    if missing:
        raise KeyError(f"{cfg.MODEL_INPUT.name} is missing columns {missing}; re-run preprocess.py")
    counts = df.groupby(cfg.YEAR_COL).size().to_dict()
    info(f"{len(df):,} schools · " + " · ".join(f"{y}: {n:,}" for y, n in counts.items()))
    return df


def versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ["pandas", "numpy", "scikit-learn", "statsmodels", "xgboost", "optuna", "shap", "pyreadstat"]:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def write_summary(sections: dict, run_info: dict, path):
    lines = [
        "# PISA × ECAIR modelling results",
        "",
        f"Generated {run_info['finished']} · cycles {', '.join(map(str, run_info['years']))} · "
        f"{run_info['n_trials']} XGBoost trials · {run_info['cf_trials']} counterfactual trials",
        "",
        "Tables follow the numbering of the September 2026 policy note. "
        "Full outputs (coefficients, models, SHAP figures, Optuna logs) are in the subfolders.",
    ]
    for title, table in sections.items():
        lines += ["", f"## {title}", "", table]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    info(f"saved [bold]{path.relative_to(cfg.ROOT)}[/bold]")


def run(args):
    unknown = [y for y in args.years if y not in cfg.CYCLES]
    if unknown:
        raise SystemExit(f"Years {unknown} are not in config.CYCLES")
    if "counterfactuals" in args.stages and cfg.CF_YEAR not in args.years:
        raise SystemExit(f"Counterfactuals use CF_YEAR={cfg.CF_YEAR}, which is not in --years")

    t_start = time.perf_counter()
    started = datetime.now().isoformat(timespec="seconds")
    out_dir = cfg.MODEL_OUTPUT_DIR
    run_pre = args.preprocess == "always" or (args.preprocess == "auto" and needs_preprocessing(args.years))
    need_escs = "econometrics" in args.stages
    n_steps = 1 + run_pre + need_escs + len(args.stages)
    set_total_steps(n_steps)

    console.print(f"[bold]PISA × ECAIR pipeline · cycles {', '.join(map(str, args.years))} · "
                  f"stages {', '.join(args.stages)}[/bold]")
    timings = {}

    if run_pre:
        import preprocess
        with step("Preprocessing (src/preprocess.py)"):
            t0 = time.perf_counter()
            preprocess.run(args.years)
            timings["preprocess"] = time.perf_counter() - t0
    elif args.preprocess == "skip" and not cfg.MODEL_INPUT.exists():
        raise SystemExit(f"{cfg.MODEL_INPUT} not found; run without --preprocess skip")

    escs = None
    if need_escs:
        import escs as escs_mod
        with step("School-mean ESCS (control for the regressions)"):
            escs = escs_mod.build_school_escs(args.years, rebuild=args.rebuild_escs)

    with step("Load modelling data"):
        df = load_model_data(args.years, escs)

    sections = {}

    if "econometrics" in args.stages:
        import econometrics
        with step("Econometrics · pooled OLS and country + cycle fixed effects"):
            t0 = time.perf_counter()
            sections.update(econometrics.run(df, out_dir)["markdown"])
            timings["econometrics"] = time.perf_counter() - t0

    pooled = None
    if "ml" in args.stages:
        import ml_models
        with step("XGBoost · Optuna tuning, test-set evaluation, SHAP"):
            t0 = time.perf_counter()
            pooled = ml_models.train_pooled(df, n_trials=args.n_trials)
            per_cycle = ml_models.train_per_cycle(df, n_trials=args.n_trials)
            sections.update(ml_models.save_results(pooled, per_cycle, out_dir))
            timings["ml"] = time.perf_counter() - t0

    if "counterfactuals" in args.stages:
        import counterfactuals
        import ml_models
        with step(f"Group counterfactuals · {cfg.CF_SOURCE_COUNTRY} → benchmarks ({cfg.CF_DOMAIN} model)"):
            t0 = time.perf_counter()
            if pooled is None or cfg.CF_DOMAIN not in pooled:
                info(f"training the pooled {cfg.CF_DOMAIN} model needed for the simulations")
                pooled = ml_models.train_pooled(df, domains=[cfg.CF_DOMAIN], n_trials=args.n_trials)
            cf = counterfactuals.run(pooled[cfg.CF_DOMAIN], out_dir, args.penalty_weights, args.cf_trials)
            sections.update(cf["markdown"])
            timings["counterfactuals"] = time.perf_counter() - t0

    run_info = {
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
        "stage_seconds": {k: round(v, 1) for k, v in timings.items()},
        "years": args.years,
        "stages": args.stages,
        "n_trials": args.n_trials,
        "cf_trials": args.cf_trials,
        "penalty_weights": args.penalty_weights,
        "quick": args.quick,
        "random_state": cfg.RANDOM_STATE,
        "ols_cov_type": cfg.OLS_COV_TYPE,
        "versions": versions(),
    }
    save_json(run_info, out_dir / "run_info.json")
    write_summary(sections, run_info, out_dir / "results_summary.md")
    console.print(f"[bold green]Finished in {run_info['elapsed_seconds']:.0f}s · "
                  f"results in {out_dir.relative_to(cfg.ROOT)}/[/bold green]")


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
