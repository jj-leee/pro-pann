"""Input collection + per-outcome result display (adapted from ML-Tongue-Pred app)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import config as appcfg
import utils as util
import shap_utils as appshap
from src import config as srccfg
from src.preprocess import remove_prefix

CO_PROC_LABELS = {
    "concurrent_vhr": "Concurrent ventral hernia repair (VHR)",
    "abdominoplasty": "Concurrent abdominoplasty",
    "concurrent_liposuction": "Concurrent liposuction",
    "concurrent_breast": "Concurrent breast procedure",
    "concurrent_trunk_flap": "Concurrent trunk flap",
    "concurrent_hysterectomy": "Concurrent hysterectomy",
    "concurrent_umbilical_hernia": "Concurrent umbilical hernia repair",
    "concurrent_body_contour": "Concurrent body contouring",
    "other_concurrent_proc": "Other concurrent procedure",
}
COMORBID_LABELS = {
    "smoke": "Current smoker (within 1 yr)", "hxcopd": "COPD", "hxchf": "CHF",
    "hypermed": "Hypertension on meds", "ascites": "Ascites", "dialysis": "Dialysis",
    "discancr": "Disseminated cancer", "steroid": "Chronic steroid use",
    "bleeddis": "Bleeding disorder", "transfus": "Preop transfusion",
}


def _num(label, default, lo, hi, key, step=1.0):
    c1, c2 = st.columns([3, 1])
    with c2:
        na = st.checkbox("N/A", key=f"na_{key}")
    with c1:
        v = st.number_input(label, min_value=float(lo), max_value=float(hi),
                            value=float(default), step=float(step), key=key,
                            disabled=na)
    return np.nan if na else v


def get_input_data():
    pre = util.load_preprocessor()
    cols = list(pre.feature_names_in_)
    row = {}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        with st.expander("Demographics", expanded=True):
            row["age"] = _num("Age (yr)", 50, 18, 90, "age")
            row["sex"] = st.selectbox("Sex", util.SEX_OPTIONS, key="sex")
            row["race_new"] = st.selectbox("Race", util.RACE_OPTIONS, key="race")
            row["ethnicity_hispanic"] = util.yn(
                st.selectbox("Hispanic ethnicity", ["No", "Yes"], key="hisp"))
        with st.expander("Anthropometrics"):
            row["height"] = _num("Height (in)", 64, 40, 90, "ht")
            row["weight"] = _num("Weight (lb)", 180, 70, 700, "wt")
    with c2:
        with st.expander("Comorbidities", expanded=True):
            row["diabetes"] = util.DIABETES_MAP[
                st.selectbox("Diabetes", list(util.DIABETES_MAP), key="dm")]
            row["asaclas"] = util.ASA_MAP[
                st.selectbox("ASA class", list(util.ASA_MAP), index=1, key="asa")]
            row["fnstatus2"] = util.FNSTATUS_MAP[
                st.selectbox("Functional status", list(util.FNSTATUS_MAP), key="fn")]
            for k, lab in COMORBID_LABELS.items():
                if k in cols:
                    row[k] = util.yn(st.selectbox(lab, ["No", "Yes"], key=k))
    with c3:
        with st.expander("Labs (leave N/A if unknown)", expanded=True):
            row["pralbum"] = _num("Albumin (g/dL)", 4.0, 1.0, 6.0, "alb", 0.1)
            row["prwbc"] = _num("WBC (x10³)", 7.0, 1.0, 50.0, "wbc", 0.1)
            row["prhct"] = _num("Hematocrit (%)", 40.0, 15.0, 60.0, "hct", 0.1)
            row["prplate"] = _num("Platelets (x10³)", 250, 50, 1000, "plt")
            if "prcreat" in cols:
                row["prcreat"] = _num("Creatinine (mg/dL)", 0.8, 0.1, 14.0, "cr", 0.1)
            if "prsodm" in cols:
                row["prsodm"] = _num("Sodium (mmol/L)", 139, 120, 155, "na")
    with c4:
        with st.expander("Operative scope", expanded=True):
            row["acuity"] = st.selectbox("Acuity", util.ACUITY_OPTIONS, key="acu")
            row["anesthes"] = st.selectbox("Anesthesia", util.ANESTH_OPTIONS, key="an")
            row["surgspec"] = st.selectbox("Surgical specialty", util.SURGSPEC_OPTIONS, key="ss")
        with st.expander("Concurrent procedures", expanded=True):
            st.caption("Check all procedures planned alongside the panniculectomy.")
            for k, lab in CO_PROC_LABELS.items():
                if k in cols:
                    row[k] = int(st.checkbox(lab, key=k))

    # fill any pipeline column not collected (defensive)
    for c in cols:
        row.setdefault(c, np.nan if c in srccfg.NUMERIC_COLS else 0)
    input_df = pd.DataFrame({c: [row[c]] for c in cols})
    return input_df


def predict(outcome, input_df):
    pre = util.load_preprocessor()
    feats = util.load_reduced_features()
    Xt = remove_prefix(pd.DataFrame(np.array(pre.transform(input_df)),
                                    columns=pre.get_feature_names_out()))
    for c in Xt.columns:
        Xt[c] = pd.to_numeric(Xt[c], errors="coerce")
    Xr = Xt[feats]
    model = util.load_model(outcome)
    return float(model.predict_proba(Xr)[0, 1]), Xr


def show_clinical_results(display_name, outcome, input_df):
    with st.expander(f"📊 {display_name}", expanded=True):
        prob, Xr = predict(outcome, input_df)
        cat, emoji, color, idx = util.get_risk_category(prob, outcome)
        thr = util.load_bin_thresholds(outcome)
        rates = util.bin_occur_rates(outcome, thr)
        pctile, n_pop = util.percentile_vs_population(prob, outcome)

        a, b, c = st.columns(3)
        a.metric("Calibrated 30-day risk", f"{prob:.2%}")
        b.markdown(f"**Risk category**<br><span style='color:{color};font-size:1.4em'>"
                   f"{emoji} {cat}</span>", unsafe_allow_html=True)
        c.metric("Risk percentile (vs 2024 test)", f"{pctile:.0f}th")
        obs = rates[idx]
        st.markdown(
            f"In our independent 2024 test cohort (n={n_pop:,}), about "
            f"**{obs:.1%}** of patients in the **{cat}** risk band developed this outcome.")

        # per-patient SHAP
        try:
            expl = util.load_explainer(outcome)
            names, phi = appshap.compute_shap(expl, Xr, key=f"{outcome}_{hash(Xr.to_json())}")
            top_n = st.slider("Top features to show", 5, min(15, len(names)), 10,
                              key=f"sh_{outcome}")
            st.pyplot(appshap.shap_plot(names, phi, top_n))
        except Exception as e:
            st.caption(f"(SHAP explanation unavailable: {e})")
