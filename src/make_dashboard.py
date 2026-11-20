"""Builds dashboard/dubai_price_dashboard.html — a single self-contained,
offline Plotly dashboard. Every number comes from results/metrics.json or
results/test_predictions.csv.gz."""
from __future__ import annotations

import html
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402


def _layout(fig, title, sub=None, height=460):
    t = f"<b>{title}</b>" + (f"<br><span style='font-size:12px;color:#5B6676'>{sub}</span>" if sub else "")
    fig.update_layout(**style.PLOTLY_LAYOUT, title=dict(text=t, x=0.01, xanchor="left"), height=height)
    return fig


def build(root: str) -> str:
    M = json.load(open(os.path.join(root, "results", "metrics.json"), encoding="utf-8"))
    te = pd.read_csv(os.path.join(root, "results", "test_predictions.csv.gz"), parse_dates=["date"])
    te["ym"] = te["date"].dt.to_period("M").astype(str)
    te["actual_psm"] = te["price_aed"] / te["area_sqm"]
    te["pred_psm"] = te["pred_price"] / te["area_sqm"]
    figs = []

    # 1. model comparison
    names = [n for n in ["Market baseline", "Comps rule (1 feature)", "Ridge (linear)", "Building rule",
                         "PerpetualBooster", "Random Forest", "XGBoost", "LightGBM"] if n in M["validation"]]
    tn = [n if n != "LightGBM" else "LightGBM (tuned, final)" for n in names]
    fig = go.Figure([
        go.Bar(name="Validation 2024", x=names, y=[M["validation"][n]["mdape_pct"] for n in names], marker_color=style.MUTED,
               text=[f"{M['validation'][n]['mdape_pct']:.1f}%" for n in names], textposition="outside"),
        go.Bar(name="Test 2025-26", x=names, y=[M["test"][n]["mdape_pct"] for n in tn], marker_color=style.ACCENT,
               text=[f"{M['test'][n]['mdape_pct']:.1f}%" for n in tn], textposition="outside"),
    ])
    wf = M["walk_forward"]["mdape_pct"]
    fig.add_trace(go.Bar(name="Test 2025-26, retrained monthly", x=["LightGBM retrained monthly"], y=[wf],
                         marker_color=style.ACCENT_2, text=[f"{wf:.1f}%"], textposition="outside"))
    fig.update_traces(textfont_size=11)
    fig.update_layout(barmode="group", bargap=0.25, yaxis_title="Median abs. % error (lower is better)", uniformtext=dict(minsize=10, mode="show"))
    figs.append(_layout(fig, "Model vs baselines",
                        "LightGBM bar on test = tuned final model; others on test are for reference only"))

    # 2. error by month
    bm = pd.DataFrame(M["test_by_month"]).T
    fig = go.Figure([
        go.Scatter(x=bm.index, y=bm["mdape_building_rule"], name="Building rule", mode="lines+markers", line=dict(color=style.MUTED, width=2)),
        go.Scatter(x=bm.index, y=bm["mdape_model"], name="Model frozen at end of 2024", mode="lines+markers", line=dict(color=style.BAD, width=2.5)),
        go.Scatter(x=bm.index, y=bm["mdape_walk_forward"], name="Model retrained monthly", mode="lines+markers", line=dict(color=style.ACCENT, width=3)),
    ])
    fig.update_layout(yaxis_title="Median abs. % error", yaxis_rangemode="tozero")
    figs.append(_layout(fig, "Error on unseen months (2025 → Nov 2026)", "A frozen model ages; retraining each month keeps it ahead of the rule"))

    # 3. results explorer by area
    top_areas = te["area_name_en"].value_counts().head(20).index.tolist()
    fig = go.Figure()
    buttons = []
    for i, ar in enumerate(top_areas):
        g = te[te.area_name_en == ar].groupby("ym").agg(actual=("actual_psm", "median"), pred=("pred_psm", "median"), n=("actual_psm", "size"))
        vis = i == 0
        fig.add_trace(go.Scatter(x=g.index, y=g["actual"], name="Actual median AED/m²", visible=vis,
                                 mode="lines+markers", line=dict(color=style.ACCENT_2, width=3),
                                 customdata=g["n"], hovertemplate="%{x}<br>actual %{y:,.0f} AED/m²<br>%{customdata} sales<extra></extra>"))
        fig.add_trace(go.Scatter(x=g.index, y=g["pred"], name="Predicted median AED/m²", visible=vis,
                                 mode="lines+markers", line=dict(color=style.ACCENT, width=3, dash="dot"),
                                 hovertemplate="%{x}<br>predicted %{y:,.0f} AED/m²<extra></extra>"))
        mask = [False] * (2 * len(top_areas)); mask[2 * i] = mask[2 * i + 1] = True
        sub = te[te.area_name_en == ar]
        md = float(np.median(np.abs(sub["signed_err_pct"])))
        buttons.append(dict(label=ar, method="update",
                            args=[{"visible": mask}, {"title.text": f"<b>Results explorer — {html.escape(ar)}</b><br><span style='font-size:12px;color:#5B6676'>{len(sub):,} test sales · median abs. error {md:.1f}%</span>"}]))
    fig.update_layout(updatemenus=[dict(buttons=buttons, direction="down", x=1.0, xanchor="right", y=1.18, yanchor="top")],
                      yaxis_title="AED per m²")
    fig.add_annotation(text=f"{M['drift']['last_month']} has only {M['drift']['last_month_is_partial_days']} days of data",
                       xref="paper", yref="paper", x=0, y=-0.2, showarrow=False, font=dict(size=11, color="#7A8494"))
    sub0 = te[te.area_name_en == top_areas[0]]
    figs.append(_layout(fig, f"Results explorer — {top_areas[0]}",
                        f"{len(sub0):,} test sales · median abs. error {np.median(np.abs(sub0['signed_err_pct'])):.1f}%", height=500))
    fig.update_layout(margin=dict(b=95))

    # 4. signed error distribution by off-plan / ready
    fig = go.Figure()
    for k, col in [("Ready", style.ACCENT_2), ("Off-Plan", style.ACCENT)]:
        e = te.loc[te.reg_type_label == k, "signed_err_pct"].clip(-60, 60)
        fig.add_trace(go.Histogram(x=e, name=k, nbinsx=120, marker_color=col, opacity=0.65, histnorm="percent"))
    fig.update_layout(barmode="overlay", xaxis_title="Predicted vs actual price (%; clipped at ±60)", yaxis_title="% of sales")
    figs.append(_layout(fig, "How far off are the predictions?", "Negative = model below the registered price"))

    # 5. area growth
    ag = pd.DataFrame(M["area_growth_2021_2025_top10"]).iloc[::-1]
    fig = go.Figure(go.Bar(x=ag["growth_pct"], y=ag["area"], orientation="h", marker_color=style.ACCENT,
                           text=[f"+{v:.0f}%" for v in ag["growth_pct"]], textposition="outside", cliponaxis=False,
                           customdata=np.c_[ag["psm_2021"], ag["psm_2025"]],
                           hovertemplate="%{y}<br>2021: %{customdata[0]:,} AED/m²<br>2025: %{customdata[1]:,} AED/m²<extra></extra>"))
    fig.update_layout(xaxis_title="Median AED/m² change 2021 → 2025", xaxis_range=[0, ag["growth_pct"].max() * 1.15])
    figs.append(_layout(fig, "Fastest-rising areas (≥300 sales in 2021 and 2025)", height=480))

    h = M["headline"]; d = M["dataset"]; iv = M["interval"]
    kpis = [
        (f"{d['rows_raw']:,}", "registered sales in the raw file"),
        (f"{h['test_mdape_pct']:.1f}%", "median error, model frozen at end of 2024"),
        (f"{h['test_within_10pct']:.1f}%", "of test sales within ±10%"),
        (f"{h['building_rule_test_mdape_pct']:.1f}%", "error of the building rule"),
        (f"{M['walk_forward']['mdape_pct']:.1f}%", "median error if retrained monthly"),
        (f"{iv['test_coverage_pct']:.1f}%", f"real coverage of the 80% interval"),
    ]
    case_rows = "".join(
        f"<tr><td>{c['case']}</td><td>{c['date']}</td><td>{html.escape(str(c['area']))}</td><td>{c['reg_type']} {c['type']}</td>"
        f"<td>{c['size_sqm']}</td><td>{c['actual_aed']:,}</td><td>{c['predicted_aed']:,}</td><td>{c['error_pct']:+.1f}%</td></tr>"
        for c in M["case_studies"])
    parts = []
    for i, f in enumerate(figs):
        parts.append(f.to_html(full_html=False, include_plotlyjs="inline" if i == 0 else False,
                               config={"displaylogo": False, "responsive": True}))
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Dubai home prices — model dashboard</title>
<style>body{{font-family:'DejaVu Sans',Arial,sans-serif;color:{style.INK};margin:0;background:#F7F8FA}}
.wrap{{max-width:1150px;margin:0 auto;padding:28px}} h1{{margin:0 0 4px}} .sub{{color:#5B6676;margin-bottom:22px}}
.kpis{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:18px}}
.kpi{{background:#fff;border-radius:10px;padding:14px;border:1px solid {style.GRID}}} .kpi b{{font-size:26px;color:{style.ACCENT}}}
.kpi span{{display:block;font-size:13px;color:#5B6676}} .card{{background:#fff;border-radius:10px;padding:8px;border:1px solid {style.GRID};margin-bottom:16px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border-bottom:1px solid {style.GRID};padding:6px;text-align:left}}
.foot{{font-size:12px;color:#7A8494}} @media(max-width:800px){{.kpis{{grid-template-columns:repeat(2,1fr)}}}}</style></head>
<body><div class="wrap"><h1>What is a Dubai home worth?</h1>
<div class="sub">Price model on official Dubai Land Department sales · trained ≤2024 · tested on {M['splits']['test']['from']} → {M['splits']['test']['to']}</div>
<div class="kpis">{''.join(f'<div class="kpi"><b>{a}</b><span>{b}</span></div>' for a, b in kpis)}</div>
{''.join(f'<div class="card">{p}</div>' for p in parts)}
<div class="card" style="padding:16px"><h3>Case studies (real test-period sales)</h3><table>
<tr><th>Case</th><th>Date</th><th>Area</th><th>Type</th><th>m²</th><th>Actual AED</th><th>Predicted AED</th><th>Error</th></tr>{case_rows}</table></div>
<p class="foot">Data: Dubai Land Department open data via dubairealestatedata.com (CC BY 4.0). Code: MIT. Built by Hussain Ahmed.</p>
</div></body></html>"""
    out = os.path.join(root, "dashboard", "dubai_price_dashboard.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write(page)
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
