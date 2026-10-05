"""App utilities: input encoders (mirror src/clean_filter), artifact loaders,
risk-category + observed-rate + percentile helpers (adapted from ML-Tongue-Pred app/utils)."""
from __future__ import annotations

import json

import joblib
import numpy as np
import streamlit as st

import config as appcfg

# ---- raw input encoders (must match src/clean_filter outputs) --------------
RACE_OPTIONS = ["White", "Black or African American", "Asian",
                "American Indian or Alaska Native",
                "Native Hawaiian or Other Pacific Islander",
                "Some Other Race", "Unknown"]
SEX_OPTIONS = ["female", "male", "non-binary"]
ACUITY_OPTIONS = ["elective", "urgent", "emergent", "Unknown"]
ANESTH_OPTIONS = ["General", "MAC/IV Sedation", "Spinal", "Local", "Epidural",
                  "Regional", "Other", "Unknown"]
SURGSPEC_OPTIONS = ["Plastics", "General Surgery", "Gynecology", "Urology",
                    "Otolaryngology (ENT)", "Orthopedics", "Vascular",
                    "Neurosurgery", "Unknown"]
DIABETES_MAP = {"No diabetes": 0, "Oral / non-insulin": 1, "Insulin-dependent": 2}
FNSTATUS_MAP = {"Independent": 0, "Partially dependent": 1, "Totally dependent": 2}
ASA_MAP = {"1 - Healthy": 1, "2 - Mild systemic disease": 2,
           "3 - Severe systemic disease": 3, "4 - Life-threatening": 4}


def yn(v):
    return 1 if v == "Yes" else 0


# ---- artifact loaders ------------------------------------------------------
@st.cache_resource
def load_preprocessor():
    return joblib.load(appcfg.PREPROC_PATH)


@st.cache_resource
def load_reduced_features():
    return joblib.load(appcfg.SHAP_DIR / "feature_names.joblib")


@st.cache_resource
def load_model(outcome):
    model_name = appcfg.CHOSEN_MODEL_DICT[outcome]
    return joblib.load(appcfg.MODELS_DIR / f"{outcome}_{model_name}.joblib")


@st.cache_resource
def load_explainer(outcome):
    return joblib.load(appcfg.SHAP_DIR / f"{outcome}.joblib")


@st.cache_data
def load_population_summary(outcome):
    """De-identified 2024 test-cohort summary (replaces the per-patient
    all_preds parquet; built by tools/build_deidentified_artifacts.py)."""
    model_name = appcfg.CHOSEN_MODEL_DICT[outcome]
    with open(appcfg.POPSUMMARY_DIR / f"{outcome}_{model_name}.json") as fh:
        return json.load(fh)


@st.cache_data
def load_bin_thresholds(outcome):
    model_name = appcfg.CHOSEN_MODEL_DICT[outcome]
    return np.load(appcfg.BINS_DIR / f"{outcome}_{model_name}.npz")["thresholds"]


# ---- risk category / bins / percentile -------------------------------------
RISK_LABELS = ["Very Low", "Low", "Moderate", "High"]
RISK_COLORS = ["#0ebd0d", "#ffd401", "#ee9410", "#c21615"]
RISK_EMOJI = ["🟢", "🟡", "🟠", "🔴"]


def get_risk_category(prob, outcome):
    thr = load_bin_thresholds(outcome)
    idx = int(np.digitize([prob], thr, right=False)[0])
    return RISK_LABELS[idx], RISK_EMOJI[idx], RISK_COLORS[idx], idx


def bin_occur_rates(outcome, thresholds):
    """Observed event rate per risk bin in the 2024 test cohort (precomputed)."""
    summ = load_population_summary(outcome)
    if not np.array_equal(np.asarray(summ["bin_thresholds"], dtype=float),
                          np.asarray(thresholds, dtype=float)):
        raise ValueError(f"bin thresholds for {outcome} do not match population summary")
    return [np.nan if r is None else float(r) for r in summ["bin_observed_rates"]]


def percentile_vs_population(prob, outcome):
    """Percentile of `prob` vs the 2024 test cohort, i.e. 100 * P(test prob < prob).

    Uses a cut-point lookup that reproduces the displayed integer percentile of
    the original per-patient computation exactly (see tools/build_deidentified_artifacts.py).
    """
    summ = load_population_summary(outcome)
    j = int(np.searchsorted(summ["percentile_cutpoints"], prob, side="left"))
    return float(summ["percentile_values"][j]), int(summ["n_test"])
