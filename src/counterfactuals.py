"""
Group counterfactuals / actionable recourse for the Philippines
(paper section 5.4.3 / 6.3, Tables 11–12, Annex D tables D1–D9).

Migrated from Version 5 (src/Version 5/PisaxCair_Modeling_modularized.ipynb, sections 4–4.4). Everything reuses the pooled XGBoost
model for CF_DOMAIN and its untouched test set; nothing is retrained here.

1. Benchmarks: mean predicted score of test-set schools in CF_YEAR, averaging
   country means so every country counts equally (ASEAN excludes the Philippines).
2. Distribution transfer: give each Philippine school the benchmark's quantile of
   every mutable feature (diagnostic scenario).
3. Group recourse: Optuna TPE searches one common package of changes to the
   Philippine group mean of each mutable feature, minimising

       max(0, T − ŷ(X + Δ)) + λ · Σ |Δ_j| / (u_j − l_j) + P

   with values clipped to the feasible range and Likert items kept integer.
4. Bootstrap uncertainty for the predicted means and gains, plus the notebook's
   scenario summary (point estimates, bootstrap CI of the mean, paired-gain CI).
"""
from __future__ import annotations

import numpy as np
import optuna
import pandas as pd

import config as cfg
from reporting import info, label, markdown_table, save_csv, task

optuna.logging.set_verbosity(optuna.logging.WARNING)

INVALID_PENALTY = 1e6


# Feasibility constraints

def permitted_ranges(X_train: pd.DataFrame, mutable: list[str], integer: list[str]) -> dict:
    lo_q, hi_q = cfg.CF_QUANTILES
    ranges = {}
    for col in mutable:
        lo, hi = float(X_train[col].quantile(lo_q)), float(X_train[col].quantile(hi_q))
        if col in cfg.CF_DOMAIN_RANGES:
            d_lo, d_hi = cfg.CF_DOMAIN_RANGES[col]
            lo, hi = max(lo, d_lo), min(hi, d_hi)
        if col in integer:
            int_lo, int_hi = float(np.ceil(lo)), float(np.floor(hi))
            lo, hi = (int_lo, int_hi) if int_lo <= int_hi else (float(np.round(lo)), float(np.round(hi)))
        ranges[col] = [lo, hi]
    return ranges


def enforce_constraints(X: pd.DataFrame, ranges: dict, integer: list[str]) -> pd.DataFrame:
    """Clip to the feasible range, then round integer-valued items and clip again."""
    X = X.copy()
    for col, (lo, hi) in ranges.items():
        if col in X.columns:
            X[col] = X[col].clip(lo, hi)
    for col in integer:
        if col in X.columns and col in ranges:
            lo, hi = ranges[col]
            X[col] = X[col].round().clip(lo, hi)
    return X


def integer_check(X_cf: pd.DataFrame, integer: list[str]) -> pd.DataFrame:
    rows = []
    for col in integer:
        vals = X_cf[col].dropna()
        max_dec = float((vals - vals.round()).abs().max()) if len(vals) else 0.0
        rows.append({"feature": col, "max_abs_decimal_part": max_dec, "passes": max_dec < 1e-9})
    return pd.DataFrame(rows)


def group_changes(X_orig: pd.DataFrame, X_cf: pd.DataFrame, mutable: list[str]) -> pd.DataFrame:
    rows = [{
        "feature": c,
        "label": label(c),
        "current_mean": X_orig[c].mean(),
        "counterfactual_mean": X_cf[c].mean(),
        "mean_change": X_cf[c].mean() - X_orig[c].mean(),
        "current_median": X_orig[c].median(),
        "counterfactual_median": X_cf[c].median(),
        "median_change": X_cf[c].median() - X_orig[c].median(),
    } for c in mutable]
    return pd.DataFrame(rows).sort_values("mean_change", key=lambda s: s.abs(), ascending=False)


# Benchmarks

def benchmark_target(model, X, meta, countries, exclude) -> dict:
    countries = [c for c in countries if c != exclude]
    available = set(meta[cfg.COUNTRY_COL].dropna().unique())
    used = [c for c in countries if c in available]
    idx = meta.index[meta[cfg.COUNTRY_COL].isin(used)]
    if len(idx) == 0:
        raise ValueError(f"No test-set schools for benchmark countries {countries}")
    pred = pd.Series(model.predict(X.loc[idx]), index=idx)
    country_means = pred.groupby(meta.loc[idx, cfg.COUNTRY_COL]).mean()
    return {
        "target_mean": float(country_means.mean()),   # countries weighted equally
        "X": X.loc[idx],
        "meta": meta.loc[idx],
        "country_means": country_means,
        "countries_used": used,
        "countries_missing": [c for c in countries if c not in available],
    }


def distribution_transfer(X_src, X_tgt, meta_tgt, mutable, ranges, integer) -> pd.DataFrame:
    """Each source school takes the benchmark's value at its own percentile rank (country-averaged)."""
    X_cf = X_src.copy()
    countries = sorted(meta_tgt[cfg.COUNTRY_COL].dropna().unique())
    for col in mutable:
        rank = X_src[col].rank(method="average", pct=True).values
        qs = []
        for c in countries:
            vals = X_tgt.loc[meta_tgt[cfg.COUNTRY_COL] == c, col].dropna().values
            if len(vals):
                qs.append(np.quantile(vals, rank))
        if qs:
            X_cf[col] = np.mean(qs, axis=0)
    return enforce_constraints(X_cf, ranges, integer)


# Group recourse

def evaluate_package(params, model, X_group, target_mean, mutable, ranges, integer,
                     penalty_weight, max_features) -> dict:
    X_cf = X_group.copy()
    selected, deltas, cost = [], {}, 0.0
    for col in mutable:
        if int(params.get(f"change__{col}", 0)) == 0:
            continue
        selected.append(col)
        delta = float(params[f"mean__{col}"]) - X_group[col].mean()
        deltas[col] = delta
        X_cf[col] = X_cf[col] + delta
        cost += abs(delta) / max(ranges[col][1] - ranges[col][0], 1e-9)

    penalty = 0.0
    if not selected:
        penalty += INVALID_PENALTY
    if len(selected) > max_features:
        penalty += INVALID_PENALTY + 1000.0 * (len(selected) - max_features)

    X_cf = enforce_constraints(X_cf, ranges, integer)
    cf_mean = float(model.predict(X_cf).mean())
    base_mean = float(model.predict(X_group).mean())
    gap = max(0.0, target_mean - cf_mean)
    return {
        "objective": gap + penalty_weight * cost + penalty,
        "cf_mean": cf_mean,
        "gain": cf_mean - base_mean,
        "changed_features": selected,
        "deltas": deltas,
        "X_cf": X_cf,
        "normalized_cost": cost,
        "target_gap": gap,
        "feature_count_penalty": penalty,
    }


def group_recourse(model, X_group, target_mean, mutable, ranges, integer, penalty_weight,
                   n_trials, seed=cfg.RANDOM_STATE, max_features=cfg.CF_MAX_FEATURES_CHANGED) -> dict:
    integer_set = set(integer)

    def objective(trial):
        params = {}
        for col in mutable:
            params[f"change__{col}"] = trial.suggest_categorical(f"change__{col}", [0, 1])
            lo, hi = ranges[col]
            if col in integer_set:
                int_lo, int_hi = int(np.ceil(lo)), int(np.floor(hi))
                if int_lo > int_hi:
                    params[f"mean__{col}"] = float(np.round(np.clip(X_group[col].mean(), lo, hi)))
                else:
                    params[f"mean__{col}"] = float(trial.suggest_int(f"mean__{col}", int_lo, int_hi))
            else:
                params[f"mean__{col}"] = trial.suggest_float(f"mean__{col}", float(lo), float(hi))
        trial.set_user_attr("params", params)
        r = evaluate_package(params, model, X_group, target_mean, mutable, ranges, integer,
                             penalty_weight, max_features)
        for k in ("cf_mean", "gain", "normalized_cost", "target_gap", "feature_count_penalty"):
            trial.set_user_attr(k, r[k])
        trial.set_user_attr("changed_features", r["changed_features"])
        return r["objective"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = evaluate_package(study.best_trial.user_attrs["params"], model, X_group, target_mean, mutable,
                            ranges, integer, penalty_weight, max_features)
    best.update(study=study, best_trial_number=study.best_trial.number)
    return best


# Bootstrap

def bootstrap_mean_ci(values, n_boot=cfg.CF_BOOTSTRAP, ci=95, seed=cfg.RANDOM_STATE) -> dict:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    boots = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)])
    a = (100 - ci) / 2
    return {"mean": v.mean(), "std": v.std(ddof=1), "boot_std": boots.std(ddof=1),
            "ci_lower": np.percentile(boots, a), "ci_upper": np.percentile(boots, 100 - a)}


def bootstrap_gain_ci(actual, cf, n_boot=cfg.CF_BOOTSTRAP, ci=95, seed=cfg.RANDOM_STATE) -> dict:
    rng = np.random.default_rng(seed)
    actual, cf = np.asarray(actual, dtype=float), np.asarray(cf, dtype=float)
    n = min(len(actual), len(cf))
    gains = cf[:n] - actual[:n]
    boots = np.array([rng.choice(gains, size=len(gains), replace=True).mean() for _ in range(n_boot)])
    a = (100 - ci) / 2
    return {"mean_gain": gains.mean(), "std_gain": gains.std(ddof=1), "boot_std_gain": boots.std(ddof=1),
            "ci_lower": np.percentile(boots, a), "ci_upper": np.percentile(boots, 100 - a)}


def scenario_rows(lam, benchmark, target_mean, baseline, preds: dict) -> list[dict]:
    """Notebook section 4.4: point estimate, bootstrap CI of the mean and paired-gain CI per scenario."""
    actual = next(iter(preds.values()))
    rows = []
    for scenario, pred in preds.items():
        b = bootstrap_mean_ci(pred)
        row = {"penalty_weight": lam, "benchmark": benchmark, "scenario": scenario,
               "predicted_mean_score": float(np.mean(pred)), "target_mean": target_mean,
               "gain_vs_actual": float(np.mean(pred)) - baseline,
               "remaining_gap_to_target": target_mean - float(np.mean(pred)),
               "std": b["std"], "boot_std": b["boot_std"], "ci_lower": b["ci_lower"], "ci_upper": b["ci_upper"]}
        if pred is not actual:
            g = bootstrap_gain_ci(actual, pred)
            row.update(std_gain=g["std_gain"], boot_std_gain=g["boot_std_gain"],
                       gain_ci_lower=g["ci_lower"], gain_ci_upper=g["ci_upper"])
        rows.append(row)
    rows.append({"penalty_weight": lam, "benchmark": benchmark, "scenario": f"{benchmark} target",
                 "predicted_mean_score": target_mean, "target_mean": target_mean,
                 "gain_vs_actual": target_mean - baseline, "remaining_gap_to_target": 0.0})
    return rows


# Main

def run(pooled_result: dict, out_dir, penalty_weights=None, n_trials: int = cfg.CF_N_TRIALS) -> dict:
    out_dir = out_dir / "counterfactuals"
    penalty_weights = penalty_weights or cfg.CF_PENALTY_WEIGHTS
    model = pooled_result["model"]
    X_train, X_test, meta_test = pooled_result["X_train"], pooled_result["X_test"], pooled_result["meta_test"]

    mutable = [c for c in cfg.CF_MUTABLE if c in X_train.columns]
    integer = [c for c in mutable if c not in cfg.CF_CONTINUOUS]
    ranges = permitted_ranges(X_train, mutable, integer)
    save_csv(pd.DataFrame([
        {"feature": c, "label": label(c), "lower": lo, "upper": hi, "integer": c in integer}
        for c, (lo, hi) in ranges.items()
    ]), out_dir / "feasible_ranges.csv")

    # Benchmark sample: test-set schools from CF_YEAR
    in_year = meta_test.index[meta_test[cfg.YEAR_COL] == cfg.CF_YEAR]
    if len(in_year) == 0:
        raise ValueError(f"No test-set schools in CF_YEAR={cfg.CF_YEAR}")
    X_bench, meta_bench = X_test.loc[in_year], meta_test.loc[in_year]
    src_idx = meta_bench.index[meta_bench[cfg.COUNTRY_COL] == cfg.CF_SOURCE_COUNTRY]
    if len(src_idx) == 0:
        raise ValueError(f"No test-set schools for {cfg.CF_SOURCE_COUNTRY} in {cfg.CF_YEAR}")
    X_src = X_bench.loc[src_idx]
    src_pred = model.predict(X_src)
    baseline = float(src_pred.mean())
    info(f"{cfg.CF_SOURCE_COUNTRY} {cfg.CF_YEAR}: {len(X_src)} test-set schools · baseline {baseline:.2f}")

    targets, target_rows = {}, []
    for name, (countries, exclude_source) in cfg.CF_BENCHMARKS.items():
        t = benchmark_target(model, X_bench, meta_bench, countries,
                             cfg.CF_SOURCE_COUNTRY if exclude_source else None)
        targets[name] = t
        target_rows.append({"benchmark": name, "target_mean": t["target_mean"],
                            "countries_used": ", ".join(t["countries_used"]),
                            "countries_missing": ", ".join(t["countries_missing"]),
                            **{f"mean_{c}": v for c, v in t["country_means"].items()}})
        info(f"{name}: target {t['target_mean']:.2f}")
    save_csv(pd.DataFrame(target_rows), out_dir / "benchmark_targets.csv")

    # Distribution transfer (diagnostic scenario)
    dist_rows, dist_changes, dist_preds = [], [], {}
    for name, t in targets.items():
        X_cf = distribution_transfer(X_src, t["X"], t["meta"], mutable, ranges, integer)
        pred = model.predict(X_cf)
        dist_preds[name] = pred
        b, g = bootstrap_mean_ci(pred), bootstrap_gain_ci(src_pred, pred)
        dist_rows.append({"benchmark": name, "baseline": baseline, "target_mean": t["target_mean"],
                          "predicted_mean": pred.mean(), "gain": pred.mean() - baseline,
                          "remaining_gap": t["target_mean"] - pred.mean(),
                          "sd_gain": g["std_gain"], "ci_lower": b["ci_lower"], "ci_upper": b["ci_upper"]})
        dist_changes.append(group_changes(X_src, X_cf, mutable).assign(benchmark=name))
    save_csv(pd.DataFrame(dist_rows), out_dir / "distribution_transfer_outcomes.csv")
    save_csv(pd.concat(dist_changes, ignore_index=True), out_dir / "distribution_transfer_changes.csv")

    # Group recourse for every λ and benchmark
    all_interventions, all_outcomes, all_boot, all_checks, all_scenarios = [], [], [], [], []
    markdown = {}
    for lam in penalty_weights:
        for name, t in targets.items():
            with task(f"λ={lam:g} · {name}: Optuna TPE {n_trials} trials"):
                best = group_recourse(model, X_src, t["target_mean"], mutable, ranges, integer, lam, n_trials)
            changes = group_changes(X_src, best["X_cf"], mutable)
            changes = changes[changes["mean_change"].abs() > 1e-9]
            all_interventions.append(changes.assign(penalty_weight=lam, benchmark=name))

            pred = model.predict(best["X_cf"])
            b, g = bootstrap_mean_ci(pred), bootstrap_gain_ci(src_pred, pred)
            gap = t["target_mean"] - baseline
            all_outcomes.append({
                "penalty_weight": lam, "benchmark": name, "baseline": baseline,
                "benchmark_score": t["target_mean"], "simulated_score": best["cf_mean"],
                "predicted_improvement": best["gain"], "remaining_gap": t["target_mean"] - best["cf_mean"],
                "benchmark_attainment_pct": 100 * (best["cf_mean"] - baseline) / gap if gap else np.nan,
                "n_interventions": len(changes), "normalized_cost": best["normalized_cost"],
                "objective": best["objective"], "best_trial": best["best_trial_number"],
            })
            all_boot.append({
                "penalty_weight": lam, "benchmark": name, "current_mean": baseline,
                "predicted_mean_after": b["mean"], "mean_gain": g["mean_gain"], "sd_gain": g["std_gain"],
                "ci_lower": b["ci_lower"], "ci_upper": b["ci_upper"],
                "gain_ci_lower": g["ci_lower"], "gain_ci_upper": g["ci_upper"],
            })
            all_checks.append(integer_check(best["X_cf"], integer).assign(penalty_weight=lam, benchmark=name))
            all_scenarios += scenario_rows(lam, name, t["target_mean"], baseline, {
                f"{cfg.CF_SOURCE_COUNTRY} actual prediction": src_pred,
                f"{cfg.CF_SOURCE_COUNTRY} with {name}-like mutable feature distribution": dist_preds[name],
                f"{cfg.CF_SOURCE_COUNTRY} group actionable recourse toward {name}": pred,
            })
            save_csv(best["study"].trials_dataframe(), out_dir / "optuna_trials" /
                     f"lambda_{lam:g}_{name.split(' ')[0].lower()}.csv", quiet=True)
            info(f"λ={lam:g} {name}: simulated {best['cf_mean']:.2f} "
                 f"(+{best['gain']:.2f}), {len(changes)} interventions")

    interventions = pd.concat(all_interventions, ignore_index=True)
    interventions = interventions[["penalty_weight", "benchmark", "feature", "label", "current_mean",
                                   "counterfactual_mean", "mean_change", "current_median",
                                   "counterfactual_median", "median_change"]]
    outcomes = pd.DataFrame(all_outcomes)
    boot = pd.DataFrame(all_boot)
    checks = pd.concat(all_checks, ignore_index=True)
    scenarios = pd.DataFrame(all_scenarios)
    if not checks["passes"].all():
        raise AssertionError("Integer constraint violated in a counterfactual; see integer_checks.csv")

    save_csv(interventions, out_dir / "interventions_all.csv")
    save_csv(outcomes, out_dir / "outcomes_all.csv")
    save_csv(boot, out_dir / "bootstrap_all.csv")
    save_csv(scenarios, out_dir / "scenario_summary_all.csv")
    save_csv(checks, out_dir / "integer_checks.csv", quiet=True)

    for lam in penalty_weights:
        tag = f"{lam:g}"
        iv = interventions[interventions["penalty_weight"] == lam]
        oc = outcomes[outcomes["penalty_weight"] == lam]
        bt = boot[boot["penalty_weight"] == lam]
        save_csv(iv, out_dir / f"lambda_{tag}" / "interventions.csv", quiet=True)
        save_csv(oc, out_dir / f"lambda_{tag}" / "predicted_outcomes.csv", quiet=True)
        save_csv(bt, out_dir / f"lambda_{tag}" / "bootstrap.csv", quiet=True)
        save_csv(scenarios[scenarios["penalty_weight"] == lam], out_dir / f"lambda_{tag}" / "scenario_summary.csv",
                 quiet=True)
        # The paper reports medians for integer (Likert/count) items and means for ratios.
        markdown[f"Interventions (λ = {tag}) · Table 11 / Annex D"] = markdown_table(
            iv[["benchmark", "label", "current_mean", "counterfactual_mean", "mean_change",
                "current_median", "counterfactual_median"]], "{:.2f}")
        markdown[f"Predicted outcomes (λ = {tag}) · Table 12 / Annex D"] = markdown_table(
            oc[["benchmark", "baseline", "benchmark_score", "simulated_score", "predicted_improvement",
                "remaining_gap", "benchmark_attainment_pct"]], "{:.2f}")
        markdown[f"Bootstrap validation (λ = {tag}) · Annex D"] = markdown_table(
            bt[["benchmark", "current_mean", "predicted_mean_after", "mean_gain", "sd_gain",
                "ci_lower", "ci_upper"]], "{:.2f}")
    info(f"saved per-λ tables under {out_dir.name}/lambda_*/")
    return {"interventions": interventions, "outcomes": outcomes, "bootstrap": boot,
            "scenarios": scenarios, "markdown": markdown}
