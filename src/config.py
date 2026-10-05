"""Central configuration + feature/outcome specification for PRO-PANN.

Mirrors `src/config.py` in ML-Tongue-Pred (SEED, BASE_PATH,
DEVICE) and additionally centralises the panniculectomy feature set, the unified
co-procedure flags, and the multi-outcome definitions described in
`SPEC_PRO-PANN_FINAL.md`. Keeping these here (rather than scattered through
notebooks, as in the siblings) lets every deliverable script import one source of
truth.
"""
from pathlib import Path

# --- Reproducibility (logged everywhere; see acceptance check 9) -------------
SEED = 42424242  # fixed project seed

# --- Paths -------------------------------------------------------------------
# Project root (the folder that contains this `src/`).
BASE_PATH = Path(__file__).resolve().parents[1]
# The pre-built cohort lives one directory up (the shared project folder).
COHORT_CSV = BASE_PATH.parent / "panniculectomy_cohort.csv"

DEVICE = "cpu"  # PRO-TONGUE also used CPU; no CUDA assumed for this build.

# --- Temporal split (strictly by cohort_year, never reshuffled) --------------
# train 2014-2021 / val 2022-2023 / test 2024  (22,931 / 5,574 / 2,829)
TRAIN_YEARS = list(range(2014, 2022))   # 2014-2021
VAL_YEARS = [2022, 2023]
TEST_YEARS = [2024]

# =============================================================================
#  FEATURE SET  (SPEC sec. 2 -- preoperative-only)
# =============================================================================
# Continuous / numerical. HEIGHT + WEIGHT are consumed into BMI by the pipeline
# (BMICalculatorArray) and then dropped, as in ML-Tongue-Pred.
NUMERIC_COLS = [
    "age", "height", "weight",
    "pralbum", "prwbc", "prhct", "prplate", "prcreat", "prsodm",
]

# Ordinal predictors (encoded to ordered ints in clean_filter, then median-imputed
# + rounded + clipped in the pipeline -- the framework's ASA handling, generalised).
# value -> (min, max) valid range after encoding
ORDINAL_COLS = ["asaclas", "diabetes", "fnstatus2"]
ORDINAL_RANGES = {
    "asaclas": (1, 4),     # 1-No Disturb ... 4/5 -> 4
    "diabetes": (0, 2),    # NO < NON-INSULIN < INSULIN
    "fnstatus2": (0, 2),   # Independent < Partially Dependent < Totally Dependent
}

# Non-ordered categoricals (one-hot, handle_unknown="ignore").
NOMINAL_COLS = ["sex", "race_new", "acuity", "anesthes", "surgspec"]

# Binary 0/1 comorbidity flags (passthrough).
BINARY_COMORBID_COLS = [
    "ethnicity_hispanic", "smoke", "hxcopd", "hxchf", "hypermed", "ascites",
    "dialysis", "discancr", "steroid", "bleeddis", "transfus",
]

# Co-procedure binary flags (already computed in the CSV -- PRO-TONGUE style).
# `concurrent_vhr` is the narrative hook + a headline SHAP result, but it is an
# ordinary predictor: NO exposure split, NO calculator toggle (SPEC sec. 2).
COPROC_COLS = [
    "concurrent_vhr", "abdominoplasty", "concurrent_liposuction",
    "concurrent_breast", "concurrent_trunk_flap", "concurrent_hysterectomy",
    "concurrent_umbilical_hernia", "concurrent_body_contour",
    "other_concurrent_proc",
]

BINARY_COLS = BINARY_COMORBID_COLS + COPROC_COLS

# Full ordered feature list fed to the preprocessor.
FEATURE_COLS = NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS + BINARY_COLS

# Dropped by the NSQIP year-availability audit (whole-year-missing across the
# entire 2021-2024 val+test era; see SPEC sec. 2 "year-availability audit").
# Confirmed empirically: dyspnea/wtloss are 100% missing for every year >= 2021.
YEAR_AUDIT_DROP = ["dyspnea", "wtloss"]

# Hard-excluded from any feature matrix (SPEC sec. 2 / acceptance check 1):
#   - optime, wndclas              -> not preoperatively available
#   - tothlos, dischdest, yrdeath  -> outcomes / leakage
#   - prsepis                      -> ~99% missing every year
#   - ind_*/vhr_old/vhr_new        -> drift descriptors / coproc components
EXCLUDED_COLS = [
    "optime", "wndclas", "tothlos", "dischdest", "yrdeath", "prsepis",
    "vhr_old", "vhr_new",
    "ind_excess_skin", "ind_panniculitis", "ind_bariatric", "ind_obesity",
]

# =============================================================================
#  OUTCOMES  (SPEC sec. 3 -- wound-dominated)
# =============================================================================
# Raw NSQIP complication columns are text-coded ("No Complication" vs the event
# name). clean_filter binarises: (notna AND != "No Complication") -> 1.
WOUND_COLS = ["supinfec", "wndinfd", "orgspcssi", "dehis"]
VTE_COLS = ["othdvt", "pulembol"]
MEDICAL_COLS = [
    "oupneumo", "reintub", "failwean", "renainsf", "oprenafl", "urninfec",
    "cnscva", "cdarrest", "cdmi", "othsysep", "othseshock",
]
# "Serious" subset (NSQIP-style severe morbidity): excludes superficial SSI, UTI,
# and isolated transfusion.
SERIOUS_COLS = [
    "wndinfd", "orgspcssi", "dehis",
    "oupneumo", "reintub", "failwean", "renainsf", "oprenafl",
    "cnscva", "cdarrest", "cdmi", "othsysep", "othseshock",
    "othdvt", "pulembol",
]
# "Any complication": union of all tracked adverse events (+ reop + bleed).
ANY_COLS = WOUND_COLS + VTE_COLS + MEDICAL_COLS + ["othbleed"]

# Outcome registry. key = short folder name used throughout artifacts.
#   display      : human-readable label (app sidebar / figures)
#   kind         : "single" (one binarised raw col), "composite" (.any over cols),
#                  "reop", "readmit", "mort", "nonhome" (special derivations)
#   cols         : raw columns the label is built from
#   deploy       : include in the Streamlit calculator? (mortality excluded, sec. 12)
#   primary      : SPEC sec. 3 primary outcome?
OUTCOME_SPEC = {
    "wound":     dict(display="Wound Complication (composite)", kind="composite", cols=WOUND_COLS,        deploy=True,  primary=True),
    "supinfec":  dict(display="Superficial SSI",                kind="single",    cols=["supinfec"],      deploy=True,  primary=False),
    "deep_ssi":  dict(display="Deep Incisional SSI",            kind="single",    cols=["wndinfd"],       deploy=True,  primary=False),
    "organ_ssi": dict(display="Organ/Space SSI",               kind="single",    cols=["orgspcssi"],     deploy=True,  primary=False),
    "dehis":     dict(display="Wound Dehiscence",               kind="single",    cols=["dehis"],         deploy=True,  primary=False),
    "reop":      dict(display="Unplanned Reoperation",          kind="reop",      cols=[],                deploy=True,  primary=True),
    "readmit":   dict(display="Unplanned Readmission",          kind="readmit",   cols=[],                deploy=True,  primary=True),
    "vte":       dict(display="Venous Thromboembolism",         kind="composite", cols=VTE_COLS,          deploy=True,  primary=False),
    "serious":   dict(display="Serious Complication",           kind="composite", cols=SERIOUS_COLS,      deploy=True,  primary=True),
    "any":       dict(display="Any Complication",               kind="composite", cols=ANY_COLS,          deploy=True,  primary=True),
    # Secondary (modelled where events permit; SPEC sec. 3 secondary)
    "bleed":     dict(display="Bleeding/Transfusion",           kind="single",    cols=["othbleed"],      deploy=True,  primary=True),
    "sepsis":    dict(display="Sepsis/Septic Shock",            kind="composite", cols=["othsysep", "othseshock"], deploy=True, primary=False),
    "nonhome":   dict(display="Non-home Discharge",             kind="nonhome",   cols=["dischdest"],     deploy=True,  primary=False),
    # Modelled but NOT deployed (too rare for a calculator; SPEC sec. 3 / 12)
    "mort":      dict(display="30-day Mortality",               kind="mort",      cols=["yrdeath", "dischdest"], deploy=False, primary=False),
}

# Default outcome set for a full run (ordered; flagship first).
ALL_OUTCOMES = list(OUTCOME_SPEC.keys())
DEPLOY_OUTCOMES = [k for k, v in OUTCOME_SPEC.items() if v["deploy"]]

# Deployment model menu (SPEC sec. 9): choose among LR / LightGBM / XGBoost only
# (NN & Stack excluded from deployment for simplicity + exact SHAP).
DEPLOY_MODEL_CANDIDATES = ["lr", "lgbm", "xgb"]
ALL_MODELS = ["lr", "lgbm", "xgb", "nn", "stack"]

# Risk-stratification gating (SPEC sec. 9): full 4-bin lift table only where the
# high-risk bin holds >= ~50 events in the 2,829-patient test set.
GATING_MIN_HIGH_BIN_EVENTS = 50

BIN_NAMES = ["Very Low", "Low", "Moderate", "High"]
N_BINS = 4
