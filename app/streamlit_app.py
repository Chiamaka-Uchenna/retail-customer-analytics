"""Retail customer analytics: interactive app.

Run with:  streamlit run app/streamlit_app.py
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AQUA, BLUE, ORANGE, bar, lines, load, predict, show, style

st.set_page_config(page_title="Retail customer analytics", layout="wide")
model, s = load("model.json"), load("summary.json")
gbm = next(m for m in s["metrics"] if m["model"] == "Gradient boosting")
recency_only = next(m for m in s["metrics"] if m["model"] == "Recency only")
ORDER = ["Champions", "New and promising", "At risk", "Lapsed"]
ADVICE = {"Champions": "Protect the relationship. No discount needed.",
          "New and promising": "Follow up with a second-order offer to build the habit.",
          "At risk": "Win back now, while they still remember the brand.",
          "Lapsed": "Low-cost reactivation only."}
FEATURE_NAMES = {"recency": "Days since last order", "frequency": "Orders in 12 months", "monetary": "Spend in 12 months",
                 "avg_order_value": "Average order value", "tenure": "Days since first order", "days_between_orders": "Usual gap between orders",
                 "orders_90d": "Orders in last 90 days", "spend_90d": "Spend in last 90 days", "products_per_order": "Products per order",
                 "returns": "Returns", "uk": "Based in the UK"}


def segment_of(recency, frequency, monetary):
    sm = s["segment_model"]
    z = [(math.log1p(v) - sm["mean"][i]) / sm["scale"][i] for i, v in enumerate([recency, frequency, monetary])]
    distances = [sum((c - zi) ** 2 for c, zi in zip(centre, z)) for centre in sm["centres"]]
    return sm["names"][int(np.argmin(distances))]


st.title("Which customers will buy again?")
st.caption("Two years of real transactions from a UK online retailer: where the revenue comes from, how customers group into segments, "
           "and a model that predicts who will order in the next 90 days.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Invoice lines analysed", f"{s['lines_raw']:,}")
c2.metric("Revenue", f"£{s['revenue_total'] / 1e6:.1f}m")
c3.metric("Revenue from top 20% of customers", f"{s['concentration']['top20']:.0%}")
c4.metric("Model AUROC", f"{gbm['AUROC']:.3f}", f"recency alone {recency_only['AUROC']:.3f}", delta_color="off")

tab_score, tab_sales, tab_retention, tab_segments, tab_model = st.tabs(["Score a customer", "Revenue", "Retention", "Segments", "Model performance"])

with tab_score:
    st.subheader("Describe a customer's last 12 months")
    a, b, c = st.columns(3)
    recency = a.number_input("Days since last order", 1, 365, 45)
    tenure = b.number_input("Days since first order (max 365)", 1, 365, 300)
    frequency = c.number_input("Orders in the last 12 months", 1, 200, 4)
    monetary = a.number_input("Spend in the last 12 months (£)", 1, 500000, 1400, 50)
    orders_90d = b.number_input("Orders in the last 90 days", 0, 100, 1)
    spend_90d = c.number_input("Spend in the last 90 days (£)", 0, 200000, 350, 50)
    products = a.number_input("Different products per order", 1, 300, 18)
    returns = b.number_input("Returns in the last 12 months", 0, 50, 0)
    uk = 1 if c.selectbox("Based in", ["United Kingdom", "Another country"]) == "United Kingdom" else 0
    tenure = max(tenure, recency)
    values = {"recency": recency, "tenure": tenure, "frequency": frequency, "monetary": monetary, "avg_order_value": monetary / frequency,
              "days_between_orders": (tenure - recency) / (frequency - 1) if frequency > 1 else None,
              "orders_90d": orders_90d, "spend_90d": spend_90d, "products_per_order": products, "returns": returns, "uk": uk}
    probability = predict(model, [values[name] for name in model["features"]])
    segment = segment_of(recency, frequency, monetary)
    percentile = int(np.searchsorted(np.array(s["score_quantiles"]), probability))
    st.divider()
    left, right = st.columns([1, 2])
    left.metric("Chance of ordering in the next 90 days", f"{probability:.0%}", segment, delta_color="off")
    right.progress(float(probability))
    right.write(f"Scores higher than about {min(percentile, 100)} percent of customers in the test period.")
    right.write(f"**{segment}:** {ADVICE[segment]}")

with tab_sales:
    st.markdown("**Monthly revenue** (December 2011 is a partial month)")
    monthly = s["monthly"]
    show(bar([m["month"] for m in monthly], [m["revenue"] / 1000 for m in monthly], "", "Revenue (£ thousand)"))
    left, right = st.columns(2)
    with left:
        st.markdown("**Revenue concentration**")
        concentration = s["concentration"]
        show(lines([("Actual", list(range(len(concentration["revenue"]))), [v * 100 for v in concentration["revenue"]], BLUE, "solid")],
                   "Share of customers, best first (%)", "Share of revenue (%)", diagonal=100))
    with right:
        st.markdown("**Revenue by country**")
        countries = s["country_share"][::-1]
        show(bar([c["country"] for c in countries], [c["share"] * 100 for c in countries], "Share of revenue (%)", "", horizontal=True))
    st.markdown("**Data cleaning audit**")
    st.dataframe(pd.DataFrame(s["cleaning"]), hide_index=True, width="stretch")

with tab_retention:
    st.caption("Each row is the group of customers who first bought in that month. Each cell is the share of that group who ordered again that many months later.")
    retention = s["retention"]
    z = [[None if v is None else v * 100 for v in row] for row in retention["values"]]
    figure = go.Figure(go.Heatmap(z=z, x=[str(a) for a in retention["ages"]], y=retention["cohorts"], colorscale="Blues", zmin=0, zmax=45,
                                  texttemplate="%{z:.0f}", xgap=2, ygap=2, colorbar=dict(title="%")))
    figure.update_xaxes(title="Months since first purchase", type="category")
    figure.update_yaxes(autorange="reversed", type="category")
    show(style(figure, 520))
    st.write(f"{s['one_time_share']:.0%} of customers placed exactly one order in two years.")

with tab_segments:
    segments = {row["segment"]: row for row in s["segments"]}
    left, right = st.columns(2)
    with left:
        st.markdown("**Share of customers against share of revenue**")
        figure = go.Figure([go.Bar(x=ORDER, y=[segments[n]["share_of_customers"] * 100 for n in ORDER], name="Share of customers", marker_color=BLUE),
                            go.Bar(x=ORDER, y=[segments[n]["share_of_revenue"] * 100 for n in ORDER], name="Share of revenue", marker_color=ORANGE)])
        figure.update_yaxes(title="%")
        show(style(figure, barmode="group"))
    with right:
        st.markdown("**Segment profiles** (median per customer)")
        table = pd.DataFrame([{"Segment": n, "Customers": segments[n]["customers"], "Days since last order": segments[n]["median_recency_days"],
                               "Orders": segments[n]["median_orders"], "Spend (£)": segments[n]["median_spend"], "What to do": ADVICE[n]} for n in ORDER])
        st.dataframe(table, hide_index=True, width="stretch")

with tab_model:
    st.caption("Trained on three earlier periods and tested on customers as of 1 September 2011, predicting orders in the following 90 days.")
    left, right = st.columns(2)
    gains = s["gains"]
    contacted = [v * 100 for v in gains["contacted"]]
    with left:
        st.markdown("**Gains chart**")
        show(lines([("Buyers, model", contacted, [v * 100 for v in gains["buyers"]], BLUE, "solid"),
                    ("Buyers, recency only", contacted, [v * 100 for v in gains["recency"]], ORANGE, "solid"),
                    ("Revenue, model", contacted, [v * 100 for v in gains["revenue"]], AQUA, "solid")],
                   "Share of customers contacted, highest score first (%)", "Share captured (%)", diagonal=100))
    with right:
        st.markdown("**Predicted against actual**")
        cal = s["calibration"]
        show(lines([("Model", [v * 100 for v in cal["predicted"]], [v * 100 for v in cal["observed"]], BLUE, "solid")],
                   "Predicted probability (%)", "Share who bought (%)", diagonal=100))
    left, right = st.columns(2)
    with left:
        st.markdown("**What drives the prediction**")
        importance = s["importance"][::-1]
        show(bar([FEATURE_NAMES.get(r["feature"], r["feature"]) for r in importance], [r["value"] for r in importance], "Mean absolute SHAP value", "", horizontal=True))
    with right:
        st.markdown("**Model comparison, out-of-time test**")
        st.dataframe(pd.DataFrame(s["metrics"]).round(3), hide_index=True, width="stretch")
