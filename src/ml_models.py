"""
XGBoost + Optuna + SHAP (paper section 5.4.2 / 6.2, Tables 7, 9, 10, Figures 5–7).

Migrated from Version 5
(src/Version 5/PisaxCair_Modeling_modularized.ipynb, the "[Results Used]" sections).

Validation design (unchanged from Version 5):
1. Hold out 20% of schools, stratified by country × PISA cycle.
2. Tune hyperparameters with Optuna (TPE) on the 80% training set only, using
   stratified 3-fold CV and a median imputer fitted inside each fold.
3. Fit the final imputer and model on the training set, evaluate once on the test set.
4. SHAP values from XGBoost's native contributions on the held-out schools.

Models:
- pooled: all cycles together, one per domain ("Merge" rows of Tables 9 and 10).
  These are the models reused by the counterfactual analysis.
- per cycle: one per cycle and domain, tuned separately with the Version 5
  per-cycle search space (2018 / 2022 rows of Tables 9 and 10).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

import config as cfg  # noqa: E402
from reporting import info, label, markdown_table, save_csv, save_json, task, warn  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

META_COLS = [cfg.COUNTRY_COL, cfg.SCHOOL_COL, cfg.YEAR_COL]


# Data

def feature_matrix(df: pd.DataFrame, target: str):
    """X (the 12 predictors), y and metadata; rows without the outcome are dropped."""
    mask = pd.to_numeric(df[target], errors="coerce").notna()
    X = df.loc[mask, cfg.FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(df.loc[mask, target], errors="coerce")
    meta = df.loc[mask, META_COLS].copy()
    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        warn(f"dropping all-missing predictors: {all_nan}")
        X = X.drop(columns=all_nan)
    return X, y, meta


def country_year_strata(meta: pd.DataFrame) -> pd.Series:
    return (
        meta[cfg.COUNTRY_COL].astype("string").fillna("MISSING_COUNTRY")
        + "__"
        + meta[cfg.YEAR_COL].astype("string").fillna("MISSING_YEAR")
    )


def impute(X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
    X_test = pd.DataFrame(imputer.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)
    return imputer, X_train, X_test


def suggest(trial: optuna.Trial, space: dict) -> dict:
    params = {}
    for name, (kind, lo, hi, opts) in space.items():
        if kind == "int":
            params[name] = trial.suggest_int(name, lo, hi, **opts)
        else:
            params[name] = trial.suggest_float(name, lo, hi, **opts)
    return params


def fixed_params(base: dict, seed: int) -> dict:
    return {**base, "random_state": seed, "n_jobs": cfg.XGB_N_JOBS}


# Metrics and explanations

def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
        "n_train": None,
        "n_test": int(len(y_test)),
    }


def shap_values(model, X: pd.DataFrame, seed: int):
    """XGBoost native SHAP contributions (last column is the base value)."""
    X_shap = X.sample(min(cfg.MAX_SHAP_ROWS, len(X)), random_state=seed)
    contribs = model.get_booster().predict(
        xgb.DMatrix(X_shap, feature_names=list(X_shap.columns)), pred_contribs=True
    )
    shap_df = pd.DataFrame(contribs[:, :-1], columns=X_shap.columns, index=X_shap.index)
    importance = shap_df.abs().mean().sort_values(ascending=False).rename("mean_abs_shap")
    return X_shap, shap_df, importance


def perm_importance(model, X_test, y_test, seed: int) -> pd.DataFrame:
    X_pfi = X_test.sample(min(2000, len(X_test)), random_state=seed)
    pfi = permutation_importance(
        model, X_pfi, y_test.loc[X_pfi.index], n_repeats=10, random_state=seed,
        n_jobs=1, scoring="neg_root_mean_squared_error",
    )
    return pd.DataFrame({
        "feature": X_pfi.columns,
        "importance_mean": pfi.importances_mean,
        "importance_std": pfi.importances_std,
    }).sort_values("importance_mean", ascending=False)


# Training

def tune_and_fit(X, y, meta, space: dict, base_params: dict, n_trials: int, name: str,
                 seed: int = cfg.RANDOM_STATE) -> dict:
    """Stratified hold-out split, Optuna CV tuning on train, final fit, test evaluation, SHAP."""
    strata = country_year_strata(meta)
    counts = strata.value_counts()
    if (counts < 2).any():
        raise ValueError(f"{name}: country × cycle strata with < 2 schools: {counts[counts < 2].to_dict()}")

    train_idx, test_idx = train_test_split(
        X.index, test_size=cfg.TEST_SIZE, random_state=seed, stratify=strata.loc[X.index]
    )
    X_train_raw, X_test_raw = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    meta_train, meta_test = meta.loc[train_idx], meta.loc[test_idx]

    split_audit = (
        pd.DataFrame({"stratum": strata, "split": np.where(strata.index.isin(test_idx), "test", "train")})
        .groupby(["stratum", "split"]).size().unstack(fill_value=0)
    )
    split_audit["total"] = split_audit.sum(axis=1)
    split_audit["test_pct"] = 100 * split_audit.get("test", 0) / split_audit["total"]

    train_strata = strata.loc[train_idx]
    n_folds = min(cfg.CV_FOLDS, int(train_strata.value_counts().min()))
    if n_folds < 2:
        raise ValueError(f"{name}: not enough schools per stratum for stratified CV")
    if n_folds != cfg.CV_FOLDS:
        warn(f"{name}: using {n_folds}-fold CV (smallest stratum is small)")
    cv_splits = list(
        StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X_train_raw, train_strata)
    )
    fixed = fixed_params(base_params, seed)

    def objective(trial):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb.XGBRegressor(**fixed, **suggest(trial, space))),
        ])
        scores = cross_val_score(pipe, X_train_raw, y_train, scoring="neg_root_mean_squared_error",
                                 cv=cv_splits, n_jobs=1)
        return -scores.mean()

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    with task(f"{name}: Optuna {n_trials} trials × {n_folds}-fold CV on {len(X_train_raw):,} schools"):
        study.optimize(objective, n_trials=n_trials)

    best_params = {**fixed, **study.best_params}
    imputer, X_train, X_test = impute(X_train_raw, X_test_raw)
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    metrics.update(n_train=len(X_train), cv_rmse=float(study.best_value))
    X_shap, shap_df, shap_imp = shap_values(model, X_test, seed)
    info(f"{name}: test MAE {metrics['mae']:.3f} · RMSE {metrics['rmse']:.3f} · R² {metrics['r2']:.3f}")

    return {
        "name": name,
        "model": model,
        "best_params": best_params,
        "study": study,
        "imputer": imputer,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "meta_train": meta_train,
        "meta_test": meta_test,
        "metrics": metrics,
        "X_shap": X_shap,
        "shap_df": shap_df,
        "shap_importance": shap_imp,
        "pfi": perm_importance(model, X_test, y_test, seed),
        "split_audit": split_audit,
    }


def train_pooled(df: pd.DataFrame, domains=None, n_trials: int = cfg.N_TRIALS) -> dict:
    out = {}
    for domain in domains or cfg.TARGETS:
        X, y, meta = feature_matrix(df, cfg.TARGETS[domain])
        out[domain] = tune_and_fit(X, y, meta, cfg.XGB_SPACE_POOLED, cfg.XGB_FIXED_POOLED, n_trials,
                                   name=f"pooled {domain}")
    return out


def train_per_cycle(df: pd.DataFrame, n_trials: int = cfg.N_TRIALS) -> dict:
    out = {}
    for year in sorted(df[cfg.YEAR_COL].unique()):
        sub = df[df[cfg.YEAR_COL] == year]
        for domain in cfg.TARGETS:
            X, y, meta = feature_matrix(sub, cfg.TARGETS[domain])
            out[(int(year), domain)] = tune_and_fit(X, y, meta, cfg.XGB_SPACE_CYCLE, cfg.XGB_FIXED_CYCLE,
                                                    n_trials, name=f"{year} {domain}")
    return out


# Tables and figures

def performance_table(pooled: dict, per_cycle: dict) -> pd.DataFrame:
    """Table 9: test-set MAE / RMSE / R² by cycle (and pooled 'Merge') and domain."""
    rows = []
    groups = [(str(y), {d: r for (yy, d), r in per_cycle.items() if yy == y})
              for y in sorted({y for y, _ in per_cycle})]
    groups.append(("Merge", pooled))
    for cycle, results in groups:
        for metric, key in [("MAE", "mae"), ("RMSE", "rmse"), ("R²", "r2")]:
            row = {"PISA Cycle": cycle, "Metric": metric}
            for d in cfg.TARGETS:
                if d in results:
                    row[d] = round(results[d]["metrics"][key], 3)
            rows.append(row)
    return pd.DataFrame(rows)


def shap_table(pooled: dict, per_cycle: dict) -> pd.DataFrame:
    """Table 10: mean |SHAP| per predictor, domain and cycle, sorted by the cross-domain mean."""
    frames = []
    groups = [(str(y), {d: r for (yy, d), r in per_cycle.items() if yy == y})
              for y in sorted({y for y, _ in per_cycle})]
    groups.append(("Merge", pooled))
    for cycle, results in groups:
        if not results:
            continue
        t = pd.concat({d: r["shap_importance"] for d, r in results.items()}, axis=1).fillna(0)
        t = t[[d for d in cfg.TARGETS if d in t.columns]]
        t["mean_across_domains"] = t.mean(axis=1)
        t = t.sort_values("mean_across_domains", ascending=False)
        t.insert(0, "Predictor", [label(f) for f in t.index])
        t.insert(0, "Year", cycle)
        t["rank"] = range(1, len(t) + 1)
        frames.append(t.reset_index(names="feature"))
    return pd.concat(frames, ignore_index=True)


def _save_fig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def shap_plots(result: dict, title: str, path_stem, dependence: list[str] = ()):
    import shap  # imported lazily: only needed for figures

    for kind, suffix in [("dot", "beeswarm"), ("bar", "bar")]:
        plt.figure()
        shap.summary_plot(result["shap_df"].to_numpy(), result["X_shap"], plot_type=kind,
                          show=False, max_display=30)
        plt.title(title)
        _save_fig(path_stem.parent / f"{path_stem.name}_{suffix}.png")

    for feature in dependence:
        if feature not in result["X_shap"].columns:
            continue
        shap.dependence_plot(feature, result["shap_df"].to_numpy(), result["X_shap"],
                             interaction_index=None, show=False)
        plt.title(f"{title} · {feature}")
        _save_fig(path_stem.parent / f"{path_stem.name}_dependence_{feature}.png")


def save_results(pooled: dict, per_cycle: dict, out_dir) -> dict:
    out_dir = out_dir / "ml"
    table9 = performance_table(pooled, per_cycle)
    table10 = shap_table(pooled, per_cycle)
    save_csv(table9, out_dir / "table_9_xgb_performance.csv")
    save_csv(table10, out_dir / "table_10_shap_importance.csv")

    all_models = {**{("pooled", d): r for d, r in pooled.items()},
                  **{(str(y), d): r for (y, d), r in per_cycle.items()}}
    params, pfi, audits = {}, [], []
    for (cycle, domain), r in all_models.items():
        tag = f"{cycle}_{domain}"
        params[tag] = {"best_params": r["best_params"], "metrics": r["metrics"]}
        pfi.append(r["pfi"].assign(model=tag))
        audits.append(r["split_audit"].reset_index().assign(model=tag))
        (out_dir / "models").mkdir(parents=True, exist_ok=True)
        r["model"].save_model(out_dir / "models" / f"xgb_{tag}.json")
        save_csv(r["study"].trials_dataframe(), out_dir / "optuna_trials" / f"{tag}.csv", quiet=True)
        title = f"SHAP · {domain} · {'Merged' if cycle == 'pooled' else cycle}"
        try:
            dependence = cfg.SHAP_DEPENDENCE_FEATURES if cycle == "pooled" else ()
            shap_plots(r, title, out_dir / "figures" / f"shap_{tag}", dependence)
        except Exception as e:  # noqa: BLE001
            warn(f"SHAP plot for {tag} failed: {type(e).__name__}: {e}")
    save_json(params, out_dir / "best_params_and_metrics.json")
    save_csv(pd.concat(pfi, ignore_index=True), out_dir / "permutation_importance.csv")
    save_csv(pd.concat(audits, ignore_index=True), out_dir / "split_audit.csv")
    info(f"saved models, Optuna trial logs and SHAP figures under {out_dir.name}/")

    top = table10.groupby("Year", sort=False).head(6)
    return {
        "Table 9 · XGBoost test-set performance": markdown_table(table9, "{:.3f}"),
        "Table 10 · Top predictors by mean |SHAP|": markdown_table(
            top[["Year", "Predictor"] + [d for d in cfg.TARGETS]], "{:.2f}"
        ),
    }
