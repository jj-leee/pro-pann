"""App config: paths + dynamic outcome/model maps written by deployment_prep."""
from __future__ import annotations

import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
BASE_PATH = APP_DIR.parent
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))


def _load_json(name, default):
    p = APP_DIR / name
    if p.exists():
        return json.load(open(p))
    return default


# {outcome_key: deployed_model_abbrev}
CHOSEN_MODEL_DICT = _load_json("chosen_models.json", {"wound": "lgbm"})
# {display_name: outcome_key}
OUTCOMES = _load_json("outcomes.json", {"Wound Complication (composite)": "wound"})

MODELS_DIR = APP_DIR / "models" / "calibrated"
PREPROC_PATH = APP_DIR / "preprocessors" / "pipeline.joblib"
# De-identified test-cohort summaries (bin observed rates + percentile lookup);
# replaces the per-patient all_preds/*.parquet, which must not be redistributed.
POPSUMMARY_DIR = APP_DIR / "population_summary"
BINS_DIR = APP_DIR / "bin_thresholds"
SHAP_DIR = APP_DIR / "shap_explainers"
