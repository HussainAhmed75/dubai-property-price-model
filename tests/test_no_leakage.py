"""Proves that no feature uses information from the same day or the future.

Run:  pytest -q tests/test_no_leakage.py
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from features import FEATURES, NUMERIC, TARGET, build_features, clean, find_data_file, load_raw  # noqa: E402
from modeling import time_split  # noqa: E402


@pytest.fixture(scope="module")
def data():
    path = find_data_file([os.path.join(ROOT, "data", "dld_sales_transactions.parquet")])
    raw = load_raw(path)
    c, _ = clean(raw)
    # a 3-year slice keeps the test fast but still has plenty of history
    c = c[(c["date"] >= "2019-01-01") & (c["date"] < "2022-01-01")].reset_index(drop=True)
    return c, build_features(c)


def _brute(c, row, key_cols, days):
    m = np.ones(len(c), bool)
    for k in key_cols:
        m &= (c[k] == row[k]).to_numpy()
    m &= (c["date"] < row["date"]).to_numpy()
    if days is not None:
        m &= (c["date"] >= row["date"] - pd.Timedelta(days=days)).to_numpy()
    v = c.loc[m, TARGET]
    return (v.mean() if len(v) else np.nan), len(v)


def test_rolling_features_match_bruteforce_past_only(data):
    c, f = data
    c = c.assign(proj_key=f["proj_key"], bldg_key=f["bldg_key"], all_key=0)
    rng = np.random.RandomState(0)
    idx = rng.choice(np.where(c["date"] >= "2020-06-01")[0], size=150, replace=False)
    checks = [("area_mean_365", "area_n_365", ["area_id"], 365),
              ("seg_mean_365", "seg_n_365", ["area_id", "property_type_label", "reg_type_label"], 365),
              ("proj_mean_all", "proj_n_all", ["proj_key"], None),
              ("bldg_mean_all", "bldg_n_all", ["bldg_key"], None),
              ("mkt_mean_30", None, ["all_key"], 30)]
    for i in idx:
        row = c.iloc[i]
        for feat, nfeat, keys, days in checks:
            if feat.startswith("proj") and row["proj_key"] == "__none__":
                continue
            if feat.startswith("bldg") and row["bldg_key"] == "__none__":
                continue
            exp_mean, exp_n = _brute(c, row, keys, days)
            got = f.iloc[i][feat]
            if exp_n == 0:
                assert np.isnan(got), (feat, i)
            else:
                assert np.isclose(got, exp_mean, atol=1e-9), (feat, i, got, exp_mean)
            if nfeat:
                assert f.iloc[i][nfeat] == exp_n, (nfeat, i)


def test_future_and_same_day_targets_cannot_change_features(data):
    """Scramble every target from a cutoff day onward; all features of rows
    registered ON or BEFORE the cutoff day must be identical."""
    c, f = data
    cutoff = pd.Timestamp("2020-07-01")
    c2 = c.copy()
    fut = c2["date"] >= cutoff
    rng = np.random.RandomState(1)
    c2.loc[fut, TARGET] = rng.normal(12, 3, fut.sum())
    f2 = build_features(c2)
    keep = (c["date"] <= cutoff).to_numpy()  # includes the cutoff day itself
    a = f.loc[keep, NUMERIC].to_numpy(float)
    b = f2.loc[keep, NUMERIC].to_numpy(float)
    assert np.allclose(a, b, equal_nan=True)
    # sanity: features AFTER the cutoff day do change
    after = (c["date"] > cutoff + pd.Timedelta(days=40)).to_numpy()
    assert not np.allclose(f.loc[after, "mkt_mean_30"], f2.loc[after, "mkt_mean_30"])


def test_target_not_in_features():
    banned = {TARGET, "psm", "price_aed", "price_psf", "quality_flags"}
    assert not banned & set(FEATURES)


def test_time_split_is_ordered(data):
    _, f = data
    g = f.copy()
    g["date"] = g["date"] + pd.DateOffset(years=4)  # shift slice into 2023-2025
    tr, va, te = time_split(g)
    assert tr["date"].max() < va["date"].min()
    assert va["date"].max() < te["date"].min()
