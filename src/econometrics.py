"""
School-level OLS: pooled baseline and country + cycle fixed effects.

Paper section 5.4.1 / 6.1, Table 8 (FE), Annex B (pooled) and Annex C (full FE).

    Score_ict = β0 + β·X_ict + γ·ESCS_ict + α_c + δ_t + ε_ict

Migrated from "[RESULTS USED] OLS w/ student data" in Version 5
(student_school_modeling_pisa_school_level_ols.ipynb).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

import config as cfg
from reporting import info, label, markdown_table, save_csv, stars, warn

SPECS = {"pooled": False, "fe": True}   # name -> add country & cycle fixed effects


def design_matrix(df: pd.DataFrame, target: str, fixed_effects: bool):
    """Predictors + ESCS (+ country and cycle dummies); rows with any missing value dropped."""
    predictors = cfg.FEATURES + [cfg.ESCS_COL]
    data = df[[cfg.COUNTRY_COL, cfg.SCHOOL_COL, cfg.YEAR_COL, target] + predictors].copy()
    data[predictors + [target]] = (
        data[predictors + [target]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    )
    data = data.dropna(subset=predictors + [target])

    X = data[predictors].astype(float)
    fe_cols: list[str] = []
    if fixed_effects:
        dummies = [
            pd.get_dummies(data[cfg.COUNTRY_COL], prefix=cfg.COUNTRY_COL, drop_first=True, dtype=float),
            pd.get_dummies(data[cfg.YEAR_COL], prefix=cfg.YEAR_COL, drop_first=True, dtype=float),
        ]
        X = pd.concat([X] + dummies, axis=1)
        fe_cols = [c for d in dummies for c in d.columns]
    return data, X, data[target].astype(float), fe_cols


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    Xv = sm.add_constant(X, has_constant="add")
    rows = []
    for i, col in enumerate(Xv.columns):
        if col == "const":
            continue
        try:
            rows.append((col, variance_inflation_factor(Xv.values, i)))
        except Exception:  # noqa: BLE001
            rows.append((col, np.nan))
    return pd.DataFrame(rows, columns=["feature", "VIF"]).sort_values("VIF", ascending=False)


def diagnostics(res, X_sm: pd.DataFrame) -> dict:
    """Specification tests run in Version 5 (RESET, Breusch–Pagan, White, Jarque–Bera)."""
    out = {}
    resid = res.resid
    tests = {
        "ramsey_reset": lambda: linear_reset(res, power=2, use_f=True),
        "breusch_pagan": lambda: het_breuschpagan(resid, X_sm),
        "white": lambda: het_white(resid, X_sm),
        "jarque_bera": lambda: jarque_bera(resid),
    }
    for name, fn in tests.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = fn()
            if name == "ramsey_reset":
                out[name] = {"F": float(r.fvalue), "p": float(r.pvalue)}
            elif name in ("breusch_pagan", "white"):
                out[name] = {"LM": float(r[0]), "LM_p": float(r[1]), "F": float(r[2]), "F_p": float(r[3])}
            else:
                out[name] = {"JB": float(r[0]), "p": float(r[1])}
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    return out


def fit_ols(df: pd.DataFrame, domain: str, spec: str) -> dict:
    target = cfg.TARGETS[domain]
    data, X, y, fe_cols = design_matrix(df, target, SPECS[spec])
    X_sm = sm.add_constant(X, has_constant="add")

    rank = np.linalg.matrix_rank(X_sm.values)
    if rank < X_sm.shape[1]:
        warn(f"{domain}/{spec}: design matrix is rank deficient ({rank} < {X_sm.shape[1]})")

    if cfg.OLS_COV_TYPE == "cluster":
        res = sm.OLS(y, X_sm).fit(cov_type="cluster", cov_kwds={"groups": data[cfg.SCHOOL_COL].to_numpy()})
    else:
        res = sm.OLS(y, X_sm).fit(cov_type=cfg.OLS_COV_TYPE)

    pred = res.predict(X_sm)
    ci = res.conf_int()
    table = pd.DataFrame({
        "domain": domain,
        "spec": spec,
        "variable": res.params.index,
        "coef": res.params.values,
        "std_err": res.bse.values,
        "z": res.tvalues.values,
        "p_value": res.pvalues.values,
        "ci_low": ci[0].values,
        "ci_high": ci[1].values,
    })
    table["sig"] = table["p_value"].map(stars)
    table["is_fixed_effect"] = table["variable"].isin(fe_cols)

    metrics = {
        "domain": domain,
        "spec": spec,
        "n_obs": int(res.nobs),
        "n_schools": int(data[[cfg.COUNTRY_COL, cfg.SCHOOL_COL, cfg.YEAR_COL]].drop_duplicates().shape[0]),
        "r2": float(r2_score(y, pred)),
        "adj_r2": float(res.rsquared_adj),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "cov_type": cfg.OLS_COV_TYPE,
    }
    return {
        "results": res,
        "table": table,
        "metrics": metrics,
        "vif": compute_vif(X).assign(domain=domain, spec=spec),
        "diagnostics": diagnostics(res, X_sm),
        "summary_text": str(res.summary()),
    }


def wide_table(coefs: pd.DataFrame, spec: str, metrics: pd.DataFrame, include_escs: bool) -> pd.DataFrame:
    """Paper-style table: one row per predictor, one column per domain, coef with stars."""
    order = list(cfg.FEATURES) + ([cfg.ESCS_COL] if include_escs else [])
    sub = coefs[(coefs["spec"] == spec) & coefs["variable"].isin(order)].copy()
    sub["cell"] = sub["coef"].map(lambda v: f"{v:.4f}") + sub["sig"]
    wide = sub.pivot(index="variable", columns="domain", values="cell").reindex(order)
    wide = wide[[d for d in cfg.TARGETS if d in wide.columns]]
    wide.index = [label(v) for v in wide.index]
    m = metrics[metrics["spec"] == spec].set_index("domain")
    wide.loc["Observations"] = [str(int(m.loc[d, "n_obs"])) for d in wide.columns]
    wide.loc["R²"] = [f"{m.loc[d, 'r2']:.3f}" for d in wide.columns]
    return wide.rename_axis("Variable").reset_index()


def run(df: pd.DataFrame, out_dir) -> dict:
    out_dir = out_dir / "econometrics"
    fits = {}
    for spec in SPECS:
        for domain in cfg.TARGETS:
            fit = fit_ols(df, domain, spec)
            fits[(spec, domain)] = fit
            m = fit["metrics"]
            info(f"{spec:>6} {domain}: n={m['n_obs']:,}  R²={m['r2']:.3f}")

    coefs = pd.concat([f["table"] for f in fits.values()], ignore_index=True)
    metrics = pd.DataFrame([f["metrics"] for f in fits.values()])
    vif = pd.concat([f["vif"] for f in fits.values()], ignore_index=True)
    diag = pd.DataFrame([
        {"spec": s, "domain": d, "test": t, **v}
        for (s, d), f in fits.items() for t, v in f["diagnostics"].items()
    ])

    save_csv(coefs, out_dir / "coefficients_all.csv")
    save_csv(metrics, out_dir / "model_fit.csv")
    save_csv(vif, out_dir / "vif.csv")
    save_csv(diag, out_dir / "diagnostics.csv")

    table8 = wide_table(coefs, "fe", metrics, include_escs=False)
    tableB1 = wide_table(coefs, "pooled", metrics, include_escs=True)
    save_csv(table8, out_dir / "table_8_fixed_effects.csv")
    save_csv(tableB1, out_dir / "table_B1_pooled_ols.csv")

    # Annex C: full FE tables per domain (country/cycle dummies omitted, as in the paper)
    for domain in cfg.TARGETS:
        full = coefs[(coefs["spec"] == "fe") & (coefs["domain"] == domain) & ~coefs["is_fixed_effect"]]
        full = full.drop(columns=["domain", "spec", "is_fixed_effect"]).assign(
            variable=lambda d: d["variable"].map(lambda v: "Constant" if v == "const" else label(v))
        )
        save_csv(full, out_dir / f"table_C_fe_full_{domain}.csv", quiet=True)

    summaries = "\n\n".join(
        f"===== {spec.upper()} · {domain} =====\n{f['summary_text']}" for (spec, domain), f in fits.items()
    )
    (out_dir / "ols_summaries.txt").write_text(summaries, encoding="utf-8")
    info(f"saved full FE tables (Annex C) and statsmodels summaries to {out_dir.name}/")

    return {
        "fits": fits,
        "coefficients": coefs,
        "metrics": metrics,
        "markdown": {
            "Table 8 · Fixed-effects estimates": markdown_table(table8),
            "Table B1 · Pooled OLS estimates": markdown_table(tableB1),
        },
    }
