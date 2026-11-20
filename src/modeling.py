"""Time-based splits, baselines, models, tuning and evaluation."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from features import CATEGORICAL, FEATURES, NUMERIC, TARGET

SEED = 42
TRAIN_END = "2023-12-31"
VAL_START, VAL_END = "2024-01-01", "2024-12-31"
TEST_START = "2025-01-01"
N_THREADS = 2


# ---------------------------------------------------------------- splits
def time_split(d: pd.DataFrame, model_start: str = "2013-01-01"):
    tr = d[(d["date"] >= model_start) & (d["date"] <= TRAIN_END)]
    va = d[(d["date"] >= VAL_START) & (d["date"] <= VAL_END)]
    te = d[d["date"] >= TEST_START]
    assert tr["date"].max() < va["date"].min() <= va["date"].max() < te["date"].min()
    return tr, va, te


# ---------------------------------------------------------------- metrics
def price_metrics(y_log_true, y_log_pred) -> dict:
    """Errors expressed on the transaction price. Because area is known,
    % error on price == % error on price-per-sqm."""
    yt = np.exp(np.asarray(y_log_true, dtype=float))
    yp = np.exp(np.asarray(y_log_pred, dtype=float))
    ape = np.abs(yp - yt) / yt
    lt, lp = np.asarray(y_log_true, float), np.asarray(y_log_pred, float)
    ss_res = np.sum((lt - lp) ** 2)
    ss_tot = np.sum((lt - lt.mean()) ** 2)
    return {
        "mdape_pct": float(np.median(ape) * 100),
        "mape_pct": float(np.mean(ape) * 100),
        "within_10pct": float(np.mean(ape <= 0.10) * 100),
        "within_20pct": float(np.mean(ape <= 0.20) * 100),
        "r2_log_psm": float(1 - ss_res / ss_tot),
        "mae_log": float(np.mean(np.abs(lt - lp))),
        "n": int(len(lt)),
    }


# ---------------------------------------------------------------- baselines
def baseline_market(d: pd.DataFrame) -> np.ndarray:
    """Naive: last-30-day market average (same registration type)."""
    return d["mkt_reg_mean_30"].fillna(d["mkt_mean_30"]).to_numpy()


def baseline_comps(d: pd.DataFrame) -> np.ndarray:
    """Agent-style 'comps' rule, ONE strong feature with fallbacks:
    same area + property type + off-plan/ready over the past 365 days,
    else whole area, else market."""
    return (d["seg_mean_365"].fillna(d["area_mean_365"])
            .fillna(d["mkt_reg_mean_30"]).fillna(d["mkt_mean_30"]).to_numpy())


def baseline_building(d: pd.DataFrame) -> np.ndarray:
    """Stronger hand-made rule: past sales in the same building, else same
    project (365d), else the comps rule."""
    return (d["bldg_mean_all"].where(d["bldg_n_all"] >= 3)
            .fillna(d["proj_mean_365"].where(d["proj_n_365"] >= 3))
            .fillna(pd.Series(baseline_comps(d), index=d.index)).to_numpy())


# ---------------------------------------------------------------- matrices
def X_tree(d: pd.DataFrame) -> pd.DataFrame:
    return d[FEATURES]


def X_codes(d: pd.DataFrame, cats_ref: pd.DataFrame) -> pd.DataFrame:
    """Integer-coded categoricals (for RandomForest)."""
    X = d[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=cats_ref[c].cat.categories).codes.astype(float)
        X.loc[X[c] < 0, c] = np.nan
    return X


# ---------------------------------------------------------------- models
def make_linear():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                    ("sc", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="ignore", min_frequency=20)
    pre = ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=1.0))])


def lgbm_params(**over):
    p = dict(objective="regression", learning_rate=0.05, n_estimators=3000,
             num_leaves=127, min_child_samples=40, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
             n_jobs=N_THREADS, deterministic=True, force_row_wise=True, verbose=-1)
    p.update(over)
    return p


def fit_lgbm(Xtr, ytr, Xva, yva, **over):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(**lgbm_params(**over))
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def fit_xgb(Xtr, ytr, Xva, yva):
    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=3000, learning_rate=0.05, max_depth=8,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                         tree_method="hist", enable_categorical=True,
                         max_cat_to_onehot=1, random_state=SEED, n_jobs=N_THREADS,
                         early_stopping_rounds=100)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


def fit_rf(Xtr, ytr, n_sub=300_000):
    from sklearn.ensemble import RandomForestRegressor
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(Xtr), size=min(n_sub, len(Xtr)), replace=False)
    m = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.5,
                              n_jobs=N_THREADS, random_state=SEED)
    m.fit(Xtr.iloc[idx], np.asarray(ytr)[idx])
    return m


def timed(fn, *a, **k):
    t = time.time()
    out = fn(*a, **k)
    return out, round(time.time() - t, 1)


# ---------------------------------------------------------------- tuning
def tune_lgbm(Xtr, ytr, Xva, yva, n_trials=25):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        over = dict(
            learning_rate=0.08,
            num_leaves=trial.suggest_int("num_leaves", 31, 511, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.05),
        )
        m = fit_lgbm(Xtr, ytr, Xva, yva, **over)
        pred = m.predict(Xva)
        trial.set_user_attr("best_iter", int(m.best_iteration_))
        return price_metrics(yva, pred)["mdape_pct"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return study


# ---------------------------------------------------------------- intervals
def split_conformal(resid_val: np.ndarray, alpha: float = 0.2) -> float:
    """Half-width (in log space) of a (1-alpha) interval from validation residuals."""
    r = np.abs(np.asarray(resid_val))
    n = len(r)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(r, min(q, 1.0)))
