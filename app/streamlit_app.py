"""Streamlit app: results explorer + what-if price simulator.

Run from the repo root:  streamlit run app/streamlit_app.py
Needs the outputs of the notebook: results/, models/.
"""
import json
import os
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *p: os.path.join(ROOT, *p)

st.set_page_config(page_title="Dubai home price model", layout="wide")


@st.cache_data
def load():
    m = json.load(open(P("results", "metrics.json"), encoding="utf-8"))
    te = pd.read_csv(P("results", "test_predictions.csv.gz"), parse_dates=["date"])
    lk = pd.read_csv(P("results", "whatif_lookup.csv"))
    meta = json.load(open(P("models", "model_meta.json")))
    return m, te, lk, meta


@st.cache_resource
def model():
    return lgb.Booster(model_file=P("models", "lgbm_production.txt"))


M, TE, LK, META = load()
BOOSTER = model()
H = M["headline"]

st.title("What is a Dubai home worth?")
st.caption(f"LightGBM trained on official DLD sales up to 2024 · tested on {M['splits']['test']['n']:,} sales "
           f"from {M['splits']['test']['from']} to {M['splits']['test']['to']}. Data: DLD via dubairealestatedata.com (CC BY 4.0).")
c = st.columns(5)
c[0].metric("Median error, frozen model", f"{H['test_mdape_pct']:.1f}%")
c[1].metric("Median error, retrained monthly", f"{M['walk_forward']['mdape_pct']:.1f}%")
c[2].metric("Within ±10% (frozen)", f"{H['test_within_10pct']:.1f}%")
c[3].metric("Building rule error", f"{H['building_rule_test_mdape_pct']:.1f}%")
c[4].metric("80% interval real coverage", f"{M['interval']['test_coverage_pct']:.1f}%")
st.caption(f"Frozen at the end of 2024, the model beats the building rule in {M['drift']['frozen_months_better_than_building_rule']} "
           f"of {M['drift']['n_months']} test months; retrained monthly, in {M['drift']['walk_forward_months_better_than_building_rule']} "
           f"of {M['drift']['n_months']}. The explorer below shows the frozen model's predictions.")

tab1, tab2 = st.tabs(["Results explorer", "What-if simulator"])

with tab1:
    f1, f2, f3 = st.columns(3)
    areas = ["All"] + TE["area_name_en"].value_counts().index.tolist()
    area = f1.selectbox("Area (official DLD name)", areas)
    reg = f2.multiselect("Off-plan / Ready", ["Off-Plan", "Ready"], default=["Off-Plan", "Ready"])
    typ = f3.multiselect("Type", ["Unit", "Villa"], default=["Unit", "Villa"])
    d = TE[TE.reg_type_label.isin(reg) & TE.property_type_label.isin(typ)]
    if area != "All":
        d = d[d.area_name_en == area]
    if d.empty:
        st.warning("No sales for this filter.")
    else:
        ape = d["signed_err_pct"].abs()
        k = st.columns(3)
        k[0].metric("Sales", f"{len(d):,}")
        k[1].metric("Median abs. error", f"{ape.median():.1f}%")
        k[2].metric("Within ±10%", f"{(ape <= 10).mean() * 100:.1f}%")
        s = d.sample(min(len(d), 5000), random_state=0)
        fig = px.scatter(s, x="price_aed", y="pred_price", color="reg_type_label", opacity=0.45, log_x=True, log_y=True,
                         color_discrete_map={"Off-Plan": "#C8963E", "Ready": "#2E7D7A"},
                         labels={"price_aed": "Actual price (AED)", "pred_price": "Predicted (AED)", "reg_type_label": ""},
                         hover_data=["date", "area_name_en", "project_name", "area_sqm", "signed_err_pct"])
        lo, hi = s[["price_aed", "pred_price"]].min().min(), s[["price_aed", "pred_price"]].max().max()
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="#1B2A41"))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(d.sort_values("date", ascending=False).head(300), width="stretch")

with tab2:
    st.write("Estimate a sale **today** with the production model (same settings, retrained on every sale "
             f"through {META['production_trained_through']}) and the latest market data in the file. "
             "Building/project history is treated as unknown — like a sale in a new building, which is the model's harder case.")
    w1, w2, w3 = st.columns(3)
    area = w1.selectbox("Area", sorted(LK["area_name_en"].unique()), key="wa")
    opts = LK[LK.area_name_en == area]
    typ = w2.selectbox("Type", sorted(opts["property_type_label"].unique()))
    regs = sorted(opts[opts.property_type_label == typ]["reg_type_label"].unique())
    reg = w3.selectbox("Off-plan / Ready", regs)
    v1, v2, v3, v4 = st.columns(4)
    size = v1.number_input("Size (m²)", 20.0, 2000.0, 80.0 if typ == "Unit" else 250.0, step=5.0)
    rooms = v2.selectbox("Bedrooms", [0, 1, 2, 3, 4, 5], index=1, format_func=lambda r: "Studio" if r == 0 else ("5+" if r == 5 else str(r)))
    parking = v3.checkbox("Parking included", True)
    proc = v4.selectbox("DLD procedure", ["Sell", "Delayed", "Development", "LeaseToOwn", "PaymentPlan", "Other", "Unknown"])

    row = opts[(opts.property_type_label == typ) & (opts.reg_type_label == reg)].iloc[0]
    rk = f"rooms_mean_365_{rooms}"
    x = {
        "log_area_sqm": np.log(size), "rooms": float(rooms), "has_parking": 1.0 if parking else 0.0,
        "month": float(date.today().month),
        "area_mean_365": row.area_mean_365, "area_n_365": row.area_n_365, "area_mean_90": row.area_mean_90,
        "seg_mean_365": row.seg_mean_365, "seg_n_365": row.seg_n_365,
        "rooms_mean_365": row[rk] if rk in row and pd.notna(row[rk]) else np.nan,
        "proj_mean_365": np.nan, "proj_n_365": 0.0, "proj_mean_all": np.nan, "proj_n_all": 0.0,
        "bldg_mean_all": np.nan, "bldg_n_all": 0.0,
        "mkt_mean_30": row.mkt_mean_30, "mkt_reg_mean_30": row.mkt_reg_mean_30, "proj_age_days": np.nan,
        "reg_type": reg, "prop_type": typ, "procedure_group": proc, "area_id_cat": str(int(row.area_id)),
    }
    X = pd.DataFrame([x])[META["features"]]
    for col in META["categorical"]:
        X[col] = pd.Categorical(X[col], categories=META["categories"][col])
    pred_log = float(BOOSTER.predict(X)[0])
    hw = META["interval_half_width_log"]
    psm = np.exp(pred_log)
    r = st.columns(3)
    r[0].metric("Estimated price", f"AED {psm * size:,.0f}")
    r[1].metric("Per m²", f"AED {psm:,.0f}")
    r[2].metric("80% range", f"AED {np.exp(pred_log - hw) * size:,.0f} – {np.exp(pred_log + hw) * size:,.0f}")
    st.caption(f"Area's past-365-day average for this segment: AED {np.exp(row.seg_mean_365):,.0f}/m² "
               f"(geometric mean of {int(row.seg_n_365):,} sales). The range was calibrated on 2024 and covered "
               f"{M['interval']['test_coverage_pct']:.1f}% of 2025-26 sales.")
