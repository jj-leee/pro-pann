# PRO-PANN: Panniculectomy 30-day Risk Calculator

A Streamlit calculator with calibrated machine-learning risk estimates and
per-patient SHAP explanations for 13 thirty-day outcomes after panniculectomy.
The models were developed on ACS-NSQIP 2014–2024: trained on 2014–2021,
calibrated on 2022–2023, and tested on the held-out 2024 cohort (n = 2,829).
30-day mortality is modelled but deliberately **not** deployed.

**Research and decision-support use only. This tool is not a medical device.**

## Acknowledgment

PRO-PANN is a fork of **[ML-Tongue-Pred](https://github.com/AnthonyMatarr/ML-Tongue-Pred)**
(PRO-TONGUE) by Anthony Matar. The Streamlit calculator, the preprocessing
pipeline, and the modeling framework (model training, calibration, lift-based
risk tiers and SHAP explanations) are adapted from that repository and applied
to panniculectomy. ML-Tongue-Pred is distributed under the MIT License. Its
copyright and permission notice are kept in [`licenses/`](licenses/).

Live app: https://pro-pann.streamlit.app (once deployed; see `DEPLOY_STEPS.md`).

## Run locally

Use Python 3.12.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/streamlit run app/base_app.py        # run from the repo root, as Streamlit Cloud does
.venv/bin/python smoke_test.py                 # headless check, prints "DEPLOY SMOKE TEST: PASS"
```

On macOS, LightGBM needs OpenMP: run `brew install libomp`.

## Layout

| Path | Contents |
|---|---|
| `app/base_app.py` | Streamlit entrypoint (`display_functions.py`, `utils.py`, `shap_utils.py`, `config.py` are its helpers) |
| `app/models/calibrated/` | 13 Platt-calibrated models (LightGBM / XGBoost / logistic regression) |
| `app/preprocessors/pipeline.joblib` | Fitted preprocessing pipeline (imputation, BMI, scaling, one-hot) |
| `app/shap_explainers/` | Per-outcome SHAP explainers and the reduced feature list |
| `app/bin_thresholds/` | Risk-category cut-points (4 bins) |
| `app/population_summary/` | De-identified 2024 test-cohort summaries: observed rate per risk bin and a percentile lookup |
| `src/` | The minimal modules the app and the pickled pipeline need (`config`, `preprocess`, `shap_utils`) |
| `tools/` | Maintenance scripts: rebuild the de-identified summaries, and check parity against the private project |
| `.streamlit/config.toml` | Theme. It must stay at the repo root for Streamlit Community Cloud |
| `packages.txt` | `libgomp1`, the OpenMP runtime LightGBM needs on Linux |

## Data privacy

This repository contains **no patient-level data**, as the ACS-NSQIP PUF data
use agreement requires:

- Per-patient test-set predictions and labels are replaced by aggregate
  summaries in `app/population_summary/*.json`. Each summary holds the observed
  event rate in each of the 4 risk bins and about 100 percentile cut-points.
  Together they reproduce the app's displayed values exactly.
- The tree SHAP explainers use `tree_path_dependent` mode and store no
  background data.
- The logistic-regression explainer (deep incisional SSI) uses a summarized
  background: a mean vector plus a covariance matrix, not training rows.
- `.gitignore` blocks `*.parquet`, `*.csv`, `*.xlsx` and `app/all_preds/`.
- `smoke_test.py` asserts that none of these files are present.

## Updating the models

1. Regenerate the artifacts in the private project.
2. Copy them into `app/`. Do not copy `app/all_preds/`.
3. Rebuild the summaries with
   `python tools/build_deidentified_artifacts.py --orig <private project>/app`.
4. Run `python smoke_test.py`, then commit and push. Community Cloud redeploys
   automatically.

Package versions in `requirements.txt` are pinned exactly to the environment
that pickled the models. Change them only together with re-exported artifacts.
