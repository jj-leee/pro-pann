"""Per-patient SHAP explanation for the app (adapted from ML-Tongue-Pred app/shap_utils)."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import shap_utils as core


PRETTY = {
    "concurrent_vhr": "Concurrent ventral hernia repair", "BMI": "BMI",
    "abdominoplasty": "Concurrent abdominoplasty",
    "concurrent_liposuction": "Concurrent liposuction",
    "concurrent_breast": "Concurrent breast procedure",
    "concurrent_hysterectomy": "Concurrent hysterectomy",
    "concurrent_umbilical_hernia": "Concurrent umbilical hernia repair",
    "concurrent_body_contour": "Concurrent body contouring",
    "concurrent_trunk_flap": "Concurrent trunk flap",
    "other_concurrent_proc": "Other concurrent procedure",
    "asaclas": "ASA class", "diabetes": "Diabetes", "age": "Age",
    "pralbum": "Preop albumin", "prwbc": "Preop WBC", "prhct": "Preop hematocrit",
    "prplate": "Preop platelets", "prcreat": "Preop creatinine",
    "prsodm": "Preop sodium", "smoke": "Current smoker", "steroid": "Chronic steroids",
    "hypermed": "Hypertension", "hxcopd": "COPD", "hxchf": "CHF",
    "bleeddis": "Bleeding disorder", "discancr": "Disseminated cancer",
    "ethnicity_hispanic": "Hispanic ethnicity", "sex": "Sex", "race_new": "Race",
    "acuity": "Acuity", "surgspec": "Surgical specialty", "anesthes": "Anesthesia",
}


def pretty(name):
    return PRETTY.get(name, name.replace("_", " ").title())


@st.cache_data(show_spinner=False)
def compute_shap(_explainer, reduced_row, key):
    """Return (feature_names, phi) with one-hot dummies recombined.

    `key` (outcome + row hash) MUST be hashed: Streamlit skips underscore-prefixed
    args when building the cache key, so with `_key` every outcome reused the first
    outcome's SHAP values for the same patient.
    """
    sv = _explainer(reduced_row, check_additivity=False)
    sv = core.combine_all_encoded(reduced_row, sv)
    vals = core.get_vals_to_plot(sv)
    return list(vals.feature_names), np.asarray(vals.values)[0]


def shap_frame(feat_names, phi, top_n=10):
    """Top-N factors (plus the remainder) as % of total |contribution|."""
    order = np.argsort(np.abs(phi))[::-1]
    top = order[:top_n]
    rest = order[top_n:]
    names = [pretty(feat_names[i]) for i in top]
    vals = [phi[i] for i in top]
    if len(rest):
        names.append(f"{len(rest)} other factors")
        vals.append(float(np.sum([phi[i] for i in rest])))
    total = np.sum(np.abs(vals)) or 1.0
    pct = [100 * v / total for v in vals]
    return pd.DataFrame({
        "factor": names,
        "pct": pct,
        "effect": ["Raises risk" if v > 0 else "Lowers risk" for v in vals],
        "label": [f"{p:+.0f}%" if round(p) != 0 else "0%" for p in pct],
    })


def shap_chart(feat_names, phi, top_n=10):
    df = shap_frame(feat_names, phi, top_n)
    lim = max(5.0, float(np.ceil(df["pct"].abs().max() * 1.3 / 5) * 5))  # room for labels
    base = alt.Chart(df).encode(
        y=alt.Y("factor:N", sort=None, title=None,
                axis=alt.Axis(labelLimit=200, labelOverlap=False, labelFontSize=12,
                              ticks=False, domain=False, labelPadding=8)),
        x=alt.X("pct:Q", title="Share of total contribution (%)",
                scale=alt.Scale(domain=[-lim, lim]),
                axis=alt.Axis(grid=True, gridColor="#eef0f3", tickCount=5)),
    )
    bars = base.mark_bar(size=16).encode(
        color=alt.Color("effect:N", legend=None,
                        scale=alt.Scale(domain=["Raises risk", "Lowers risk"],
                                        range=["#c21615", "#5b7fa6"])),
        tooltip=[alt.Tooltip("factor:N", title="Factor"),
                 alt.Tooltip("effect:N", title="Effect"),
                 alt.Tooltip("pct:Q", title="Share (%)", format="+.1f")],
    )
    text_up = base.transform_filter("datum.pct > 0").mark_text(
        align="left", dx=4, fontSize=11, color="#4b5563").encode(text="label:N")
    text_down = base.transform_filter("datum.pct <= 0").mark_text(
        align="right", dx=-4, fontSize=11, color="#4b5563").encode(text="label:N")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#9ca3af").encode(x="x:Q")
    return (bars + text_up + text_down + zero).properties(height=32 * len(df))
