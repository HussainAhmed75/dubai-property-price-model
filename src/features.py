"""Data loading, cleaning and leakage-safe feature engineering for the
Dubai DLD sales-transactions dataset.

Every "history" feature for a transaction registered on day D is computed
ONLY from transactions registered on days strictly before D (closed='left'
rolling windows on a daily aggregate). Same-day transactions never see each
other, so nothing from the future (or the present day) leaks into a row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SQM_TO_SQFT = 10.7639

# Residential sub-types we keep (others inside usage=Residential are offices,
# shops, hotel rooms ... which price on a different logic).
KEEP_SUBTYPES = {"Flat", "Villa", "Stacked Townhouses", "Unit"}

MODEL_START = "2013-01-01"   # first date used for training rows
HISTORY_START = "2010-01-01"  # earlier rows only feed the rolling history

CATEGORICAL = ["reg_type", "prop_type", "procedure_group", "area_id_cat"]

NUMERIC = [
    "log_area_sqm", "rooms", "has_parking", "month",
    "area_mean_365", "area_n_365", "area_mean_90",
    "seg_mean_365", "seg_n_365",
    "rooms_mean_365",
    "proj_mean_365", "proj_n_365", "proj_mean_all", "proj_n_all",
    "bldg_mean_all", "bldg_n_all",
    "mkt_mean_30", "mkt_reg_mean_30",
    "proj_age_days",
]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "log_psm"   # log(price per square metre, AED)


def find_data_file(candidates=None) -> str:
    """Locate the parquet file locally or under /kaggle/input."""
    import glob
    import os
    candidates = candidates or [
        "data/dld_sales_transactions.parquet",
        "../data/dld_sales_transactions.parquet",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    hits = glob.glob("/kaggle/input/**/dld_sales_transactions.parquet", recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError("dld_sales_transactions.parquet not found")


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Apply documented cleaning steps; return cleaned frame + a step log."""
    log: list[dict] = []

    def step(name, reason, new):
        log.append({"step": name, "reason": reason,
                    "rows_before": int(len(cur)), "rows_after": int(len(new)),
                    "rows_removed": int(len(cur) - len(new))})
        return new

    cur = df.copy()
    log.append({"step": "raw", "reason": "file as downloaded",
                "rows_before": int(len(cur)), "rows_after": int(len(cur)), "rows_removed": 0})
    cur["date"] = pd.to_datetime(cur["instance_date"], errors="coerce")

    cur = step("quality_flags == 0",
               "publisher's own checks: price floor, area range, P1-P99 psf band, bad dates",
               cur[cur["quality_flags"] == 0])
    cur = step("residential Unit/Villa only",
               "Land trades on plot area and whole Buildings price wholesale; commercial units price differently",
               cur[cur["property_type_label"].isin(["Unit", "Villa"]) & (cur["usage_label"] == "Residential")])
    cur = step("drop non-home sub-types",
               "offices/shops/hotel rooms/hotel apartments registered under Residential",
               cur[cur["property_sub_type"].isna() | cur["property_sub_type"].isin(KEEP_SUBTYPES)])
    dup_cols = [c for c in df.columns if c != "transaction_id"]
    cur = step("drop exact duplicates (all columns except transaction_id)",
               "cannot tell identical bulk sales from re-registrations; keep one so no single deal dominates",
               cur.drop_duplicates(dup_cols))
    cur = step(f"date >= {HISTORY_START}",
               "pre-2010 years are thin (<10k rows/yr before 2007) and only add noise to rolling history",
               cur[cur["date"] >= HISTORY_START])

    cur = cur.sort_values(["date", "transaction_id"]).reset_index(drop=True)
    cur["psm"] = cur["price_aed"] / cur["area_sqm"]
    cur[TARGET] = np.log(cur["psm"])
    return cur, log


def _procedure_group(s: pd.Series) -> pd.Series:
    s = s.fillna("Unknown")
    out = np.where(s.str.contains("Lease to Own"), "LeaseToOwn",
          np.where(s.str.contains("Development"), "Development",
          np.where(s.str.contains("Delayed"), "Delayed",
          np.where(s.str.contains("Payment Plan"), "PaymentPlan",
          np.where(s.str.startswith("Sell"), "Sell", "Other")))))
    return pd.Series(out, index=s.index)


def _past_rolling(df: pd.DataFrame, keys: list[str], window: str | None,
                  value: str = TARGET) -> pd.DataFrame:
    """Mean and count of `value` for each key over the trailing `window`
    of days STRICTLY BEFORE each row's date (window=None -> all past days).

    Implementation: aggregate to (key, day) sums/counts, then a left-closed
    time rolling window per key, then join back on (key, day).
    """
    g = (df.groupby(keys + ["date"], observed=True, sort=True)[value]
           .agg(["sum", "count"]).reset_index())
    if window is None:
        grp = g.groupby(keys, observed=True, sort=False)
        g["s"] = grp["sum"].cumsum() - g["sum"]      # exclude current day
        g["n"] = grp["count"].cumsum() - g["count"]
    else:
        parts = []
        for _, sub in g.groupby(keys, observed=True, sort=False):
            r = sub.set_index("date")[["sum", "count"]].rolling(window, closed="left").sum()
            sub = sub.copy()
            sub["s"] = r["sum"].to_numpy()
            sub["n"] = r["count"].to_numpy()
            parts.append(sub)
        g = pd.concat(parts)
    g["n"] = g["n"].fillna(0)
    g["mean"] = np.where(g["n"] > 0, g["s"] / g["n"].replace(0, np.nan), np.nan)
    out = df[keys + ["date"]].merge(g[keys + ["date", "mean", "n"]],
                                    on=keys + ["date"], how="left")
    return out[["mean", "n"]].set_axis(df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the cleaned frame with all model features added."""
    d = df.copy()
    d["reg_type"] = d["reg_type_label"].astype("category")
    d["prop_type"] = d["property_type_label"].astype("category")
    d["procedure_group"] = _procedure_group(d["procedure_name_en"]).astype("category")
    d["area_id_cat"] = d["area_id"].astype("Int64").astype(str).astype("category")
    d["log_area_sqm"] = np.log(d["area_sqm"])
    d["rooms"] = d["rooms_bucket"].astype(float)
    d["rooms_key"] = d["rooms_bucket"].fillna(-1).astype(int)
    d["has_parking"] = d["has_parking"].map({True: 1.0, False: 0.0}).astype(float)
    d["month"] = d["date"].dt.month.astype(float)
    d["proj_key"] = d["project_name"].fillna("__none__").str.strip().str.lower()
    d["bldg_key"] = d["building_name"].fillna("__none__").str.strip().str.lower()
    d["all_key"] = 0

    def put(prefix, keys, window):
        r = _past_rolling(d, keys, window)
        d[f"{prefix}_mean"] = r["mean"]
        d[f"{prefix}_n"] = r["n"]

    put("area_365", ["area_id"], "365D")
    put("area_90", ["area_id"], "90D")
    put("seg_365", ["area_id", "property_type_label", "reg_type_label"], "365D")
    put("rooms_365", ["area_id", "rooms_key"], "365D")
    put("proj_365", ["proj_key"], "365D")
    put("proj_all", ["proj_key"], None)
    put("bldg_all", ["bldg_key"], None)
    put("mkt_30", ["all_key"], "30D")
    put("mkt_reg_30", ["reg_type_label"], "30D")

    d = d.rename(columns={
        "area_365_mean": "area_mean_365", "area_365_n": "area_n_365",
        "area_90_mean": "area_mean_90",
        "seg_365_mean": "seg_mean_365", "seg_365_n": "seg_n_365",
        "rooms_365_mean": "rooms_mean_365",
        "proj_365_mean": "proj_mean_365", "proj_365_n": "proj_n_365",
        "proj_all_mean": "proj_mean_all", "proj_all_n": "proj_n_all",
        "bldg_all_mean": "bldg_mean_all", "bldg_all_n": "bldg_n_all",
        "mkt_30_mean": "mkt_mean_30", "mkt_reg_30_mean": "mkt_reg_mean_30",
    })
    # "no project / no building" is not a real group -> blank its history
    none_p = d["proj_key"] == "__none__"
    d.loc[none_p, ["proj_mean_365", "proj_mean_all"]] = np.nan
    d.loc[none_p, ["proj_n_365", "proj_n_all"]] = 0
    none_b = d["bldg_key"] == "__none__"
    d.loc[none_b, "bldg_mean_all"] = np.nan
    d.loc[none_b, "bldg_n_all"] = 0

    # project age: days since the project's first sale strictly before today
    first = (d[~none_p].groupby("proj_key")["date"].min())
    d["proj_age_days"] = (d["date"] - d["proj_key"].map(first)).dt.days.astype(float)
    d.loc[none_p, "proj_age_days"] = np.nan
    return d
