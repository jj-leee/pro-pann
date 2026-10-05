# Development, Validation, and Deployment of a Machine Learning Risk Calculator for 30-Day Complications After Panniculectomy

[![Forked from ML-Tongue-Pred](https://img.shields.io/badge/forked%20from-ML--Tongue--Pred-blue.svg)](https://github.com/AnthonyMatarr/ML-Tongue-Pred)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description
This project implements Logistic Regression, LightGBM, XGBoost, Neural Network, and Stacked Generalization models to predict 30-day postoperative complications after panniculectomy (CPT 15830) in the American College of Surgeons National Surgical Quality Improvement Program (ACS-NSQIP) dataset.

**Primary outcomes:**
- Wound complications
- Any complication
- Unplanned reoperation
- Unplanned readmission
- Bleeding requiring transfusion

**Secondary outcomes** (modeled and reported in the manuscript, but not deployed):
- Serious complications
- Superficial, deep, and organ/space surgical site infection
- Wound dehiscence
- Venous thromboembolism
- Sepsis
- Non-home discharge
- Mortality

Models are developed with data from 2014-2021, calibrated on 2022-2023, and validated on a held-out 2024 cohort.

## Building on the Framework of Matar et al.
PRO-PANN builds on the machine learning framework that Matar et al. developed for PRO-TONGUE ([ML-Tongue-Pred](https://github.com/AnthonyMatarr/ML-Tongue-Pred)).

### Fork of ML-Tongue-Pred
This repository is a GitHub fork of [ML-Tongue-Pred](https://github.com/AnthonyMatarr/ML-Tongue-Pred), the most recent implementation of the framework. It keeps ML-Tongue-Pred's commit history, and the PRO-PANN code replaces the upstream file tree.

**Carried over from the framework:**
- A preprocessing pipeline fit on training data only. It covers iterative imputation, BMI derived from height and weight, scaling, ordinal clipping, and one-hot encoding.
- Hyperparameter-tuned logistic regression, LightGBM, XGBoost, neural network, and stacked generalization models, each calibrated on a separate validation cohort.
- Lift-based risk stratification into four log-spaced risk bins. For each outcome, the deployed model is chosen by monotonic lift across the bins.
- Patient-level SHAP explanations.
- The Streamlit interface (`base_app.py`, `display_functions.py`, `utils.py`, `shap_utils.py`) and the keep-awake workflow.

**Adapted for panniculectomy:**
- **Cohort:** cases with CPT 15830 in any CPT field of ACS-NSQIP 2014-2024, split strictly by year.
- **Predictors:** 37 preoperative variables, including nine concurrent-procedure flags (e.g., ventral hernia repair, abdominoplasty, liposuction). Dyspnea and weight loss were excluded because ACS-NSQIP has not recorded them since 2021.
- **Outcomes:** 14 outcomes are modeled, and the 5 primary outcomes are deployed. They include bleeding requiring transfusion, which the ACS-NSQIP Surgical Risk Calculator does not estimate.
- **Deployment artifacts:** aggregate test-cohort summaries replace per-patient predictions (see [Data Privacy](#data-privacy)).

## Associated Risk Calculator
**PRO-PANN** (**P**ostoperative **R**isk **O**utcomes after **Pann**iculectomy) is a web application that deploys **LightGBM** models (wound complications, unplanned reoperation) and **XGBoost** models (any complication, unplanned readmission, bleeding requiring transfusion) for the five primary outcomes. It stratifies an input patient into one of **Very Low, Low, Moderate, or High** risk bins based on calibrated probability output. The interface can be found [here](https://pro-pann.streamlit.app/).

Once all [Installation Steps](#installation) are completed, you can also run the app locally with `.venv/bin/streamlit run app/base_app.py`.

### Features
- Enter the patient in four sections: Patient, Health status, Preoperative labs, and Operation (including concurrent procedures). BMI is calculated from height and weight.
- Laboratory values may be marked not available, and the fitted preprocessing pipeline imputes them.
- Choose which of the 5 primary outcomes to estimate (all are selected by default), then select **Estimate risk**.
- Results show a summary card per outcome and a detailed tab for each:
  - **Calibrated risk** probability
  - **Risk stratification** into one of Very Low, Low, Moderate, or High risk, shown on a tier scale with cut points
  - **Observed event rate** among 2024 test-cohort patients in each risk tier
  - **Percentile ranking** of model output relative to the 2024 cohort
  - **Contributing factors** via patient-level SHAP explanation values
- The sidebar summarizes the models and defines each outcome.
- Secondary outcomes are modeled but not deployed.

## Project Layout
### Included directories
- `app/`: all code and artifacts for the interface
  - `base_app.py`: source code for the calculator interface
  - `display_functions.py`: helper file for the modules displayed on the interface
  - `utils.py`: helper functions for risk bins, observed rates, and percentiles
  - `shap_utils.py`: helper functions for SHAP analysis
  - `models/calibrated/`: calibrated model for each primary outcome
  - `preprocessors/pipeline.joblib`: fitted preprocessing pipeline
  - `shap_explainers/`: per-outcome SHAP explainers and the reduced feature list
  - `bin_thresholds/`: risk-bin cut points
  - `population_summary/`: de-identified 2024 test-cohort summaries
- `src/`: the minimal modules the app and the pickled pipeline need (`config`, `preprocess`, `shap_utils`)
- `tools/`: scripts to rebuild the de-identified summaries and check parity with the private project
- `.streamlit/`: interface styling. It stays at the repository root for Streamlit Community Cloud.
- `.github/workflows/keep_awake.yml` and `keep_awake.py`: a scheduled job that keeps the hosted app awake
- `licenses/`: the original MIT notice for ML-Tongue-Pred

### Not included in this repository
- `data/`: raw and processed ACS-NSQIP data
- `models/`: raw trained and calibrated models for every candidate model type
- `results/`: figures and tables
- The analysis pipeline: cohort cleaning, preprocessing, tuning, training, risk stratification, evaluation, and drift analysis

## Installation

### Prerequisites
Ensure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed on your system.

### Steps (run all in command prompt/terminal)
1. Clone this repository with HTTPS or SSH:

- Using HTTPS (**recommended for simplicity**):
```
git clone https://github.com/jj-leee/pro-pann.git
```
- Using SSH:
```
git clone git@github.com:jj-leee/pro-pann.git
```
2. Navigate to the project directory:
```
cd pro-pann
```
3. Ensure data/object integrity and consistency:
```
git fsck --full
```
4. Create a Python 3.12 environment from the pinned requirements:
```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```
5. Run the headless check, which prints `DEPLOY SMOKE TEST: PASS`:
```
.venv/bin/python smoke_test.py
```

All paths resolve relative to the repository, so no path configuration is needed.

## Troubleshooting

### macOS: LightGBM Import Error (libomp.dylib)
If you encounter an error when importing LightGBM on macOS:
```
OSError: Library not loaded: @rpath/libomp.dylib
```
**Solution**: Install the OpenMP library using Homebrew:
```
brew install libomp
```

## Usage
- **Deployment:** the app is deployed on Streamlit Community Cloud with entrypoint `app/base_app.py` and Python 3.12. `packages.txt` installs `libgomp1`, the OpenMP runtime LightGBM needs on Linux. Full steps are in `DEPLOY_STEPS.md`.
- **Keep-awake:** Streamlit Community Cloud puts apps to sleep after inactivity. A GitHub Actions workflow opens the app every 8 hours, clicks the wake button if the app is asleep, and confirms it rendered. It can also be triggered manually from the Actions tab.
- **Pinned versions:** package versions in `requirements.txt` exactly match the environment that pickled the models. Change them only when you also re-export the artifacts.
- **Updating the models:**
  1. Regenerate the artifacts in the private project.
  2. Copy them into `app/`. Do not copy `app/all_preds/`.
  3. Run `python tools/build_deidentified_artifacts.py --orig <private project>/app`.
  4. Run `smoke_test.py`, then push.
- **Note:** a consistent random seed is used throughout the project. Even so, minor numerical deviations from the manuscript may occur because of OS/architecture differences and solver choices.

## Data Privacy
This repository contains **no patient-level data**, as the ACS-NSQIP Participant Use File data use agreement requires:
- **Test-cohort summaries:** per-patient predictions and labels are replaced by aggregate summaries in `app/population_summary/`. These hold the observed event rate per risk bin and percentile cut points, and they reproduce the displayed values exactly.
- **SHAP explainers:** all deployed explainers are tree-based and store no background data.
- **Safeguards:** `.gitignore` blocks `*.parquet`, `*.csv`, `*.xlsx`, and `app/all_preds/`. `smoke_test.py` asserts that none are present.

## License: MIT
- Code is licensed under MIT (see `LICENSE`), and the upstream copyright of Anthony Matar is retained.
- The original ML-Tongue-Pred notice is kept in `licenses/`.
- No patient data are included.

## Citation
If you use this code, please also cite the framework it builds on:

**Manuscript:** Matar DY, Matar AY, Nimbalkar A, et al. Artificial Intelligence–Based Risk Prediction Models for Complications After Tongue Cancer Surgery. *JAMA Otolaryngol Head Neck Surg.* Published online June 18, 2026. doi:10.1001/jamaoto.2026.1453

**Software:** Matar AY (2026). ML-Tongue-Pred (v1.1.0). Zenodo. https://doi.org/10.5281/zenodo.20750589

The PRO-PANN manuscript is in preparation.
