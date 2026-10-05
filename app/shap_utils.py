"""Per-patient SHAP explanation for the app (adapted from ML-Tongue-Pred app/shap_utils)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def shap_plot(feat_names, phi, top_n=10):
    order = np.argsort(np.abs(phi))[::-1]
    top = order[:top_n]
    rest = order[top_n:]
    names = [pretty(feat_names[i]) for i in top]
    vals = [phi[i] for i in top]
    if len(rest):
        names.append(f"{len(rest)} other features")
        vals.append(float(np.sum([phi[i] for i in rest])))
    total = np.sum(np.abs(vals)) or 1.0
    pct = [100 * v / total for v in vals]
    colors = ["#ff006e" if v > 0 else "#118ab2" for v in vals]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.45 * len(names))))
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, pct, color=colors)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Contribution to risk (%)  —  pink ↑ risk, blue ↓ risk")
    fig.tight_layout()
    return fig
