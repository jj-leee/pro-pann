"""PRO-PANN Streamlit risk calculator (adapted from the ML-Tongue-Pred app).

Inputs = the deployed feature set; concurrent-procedure flags are user-set
checkboxes. Per selected outcome the app shows the calibrated 30-day risk, the
log-spaced risk category, the observed in-category rate from the 2024 test set,
the outcome-specific percentile, and a per-patient SHAP explanation.

Only the five primary outcomes are deployed. There is deliberately NO
concurrent-procedure toggle/compare control (SPEC sec. 2 prohibition) -- the
co-procedures are descriptive inputs to a single prediction, never an A/B switch.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import streamlit as st
import display_functions as display


def main():
    st.set_page_config(page_title="PRO-PANN", page_icon="🏥", layout="wide")
    display.apply_style()
    display.sidebar_about()

    st.title("PRO-PANN: Postoperative Risk Outcomes after Panniculectomy", anchor=False)
    st.caption("Estimates 30-day complication risk after panniculectomy from preoperative "
               "patient and operative factors. For research and clinical decision "
               "support only.")

    input_df = display.get_input_data()

    with st.container(border=True):
        selected = display.select_outcomes()
        if st.button("Estimate risk", type="primary", disabled=not selected):
            st.session_state["go"] = True
        if not selected:
            st.caption("Select at least one outcome.")
    if not selected:
        return
    if st.session_state.get("go"):
        st.divider()
        display.show_results(selected, input_df)


if __name__ == "__main__":
    main()
