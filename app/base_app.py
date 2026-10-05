"""PRO-PANN Streamlit risk calculator (adapted from the ML-Tongue-Pred app).

Inputs = the deployed feature set; concurrent-procedure flags are user-set
checkboxes. Per selected outcome the app shows the calibrated 30-day risk, the
log-spaced risk category, the observed in-category rate from the 2024 test set,
the outcome-specific percentile, and a per-patient SHAP explanation.

Mortality is NOT deployed (SPEC sec. 12). There is deliberately NO
concurrent-procedure toggle/compare control (SPEC sec. 2 prohibition) -- the
co-procedures are descriptive inputs to a single prediction, never an A/B switch.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import streamlit as st
import display_functions as display
from config import OUTCOMES


def main():
    st.set_page_config(page_title="PRO-PANN", page_icon="🏥", layout="wide")
    st.title("PRO-PANN — Panniculectomy 30-day Risk Calculator")
    st.markdown(
        "Interpretable ML risk estimates for 30-day outcomes after panniculectomy "
        "(ACS-NSQIP 2014–2024). Models are calibrated and validated on a held-out "
        "2024 test cohort. **Research/decision-support use only.**")
    st.info("Adjust every field to match your patient. Unknown labs can be marked "
            "N/A and will be imputed by the modeling pipeline.")

    st.sidebar.header("Select outcomes to predict")
    selected = []
    for display_name, outcome in OUTCOMES.items():
        if st.sidebar.checkbox(display_name, value=(outcome == "wound")):
            selected.append((display_name, outcome))
    if not selected:
        st.warning("Select at least one outcome in the sidebar.")
        return

    input_df = display.get_input_data()

    if st.button("Predict outcomes", type="primary"):
        st.session_state["go"] = True
    if st.session_state.get("go"):
        st.header("Prediction results")
        st.caption("Co-procedure selections are inputs to a single risk estimate; "
                   "this tool does not compare 'with vs without' a concurrent procedure.")
        for display_name, outcome in selected:
            display.show_clinical_results(display_name, outcome, input_df)


if __name__ == "__main__":
    main()
