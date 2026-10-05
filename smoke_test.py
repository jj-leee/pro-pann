"""Headless smoke test for the deployable PRO-PANN app (run from the repo root).

Imports the app modules exactly as `streamlit run app/base_app.py` does, loads
EVERY runtime artifact, and for all deployed outcomes runs
preprocess -> reduce -> calibrated predict -> risk bin -> observed bin rate ->
percentile -> per-patient SHAP (+ additivity check) -> SHAP bar plot.
Also asserts that no patient-level data ships with the bundle.

    python smoke_test.py      # prints "DEPLOY SMOKE TEST: PASS"
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parent
APP = REPO / "app"
sys.path.insert(0, str(APP))

import logging  # noqa: E402

logging.getLogger("streamlit").setLevel(logging.ERROR)
caught: list[warnings.WarningMessage] = []
warnings.simplefilter("always")
_orig_showwarning = warnings.showwarning
warnings.showwarning = lambda m, c, f, l, file=None, line=None: caught.append(
    warnings.WarningMessage(m, c, f, l))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

import config as appcfg  # noqa: E402
import utils as util  # noqa: E402
import display_functions as display  # noqa: E402
import shap_utils as appshap  # noqa: E402

for _name in list(logging.root.manager.loggerDict):  # silence bare-mode ScriptRunContext noise
    if _name.startswith("streamlit"):
        logging.getLogger(_name).setLevel(logging.ERROR)


def privacy_checks():
    banned = {".parquet", ".csv", ".xlsx", ".xls", ".feather", ".sav", ".dta"}
    bad = [p for p in REPO.rglob("*") if p.suffix.lower() in banned
           and ".venv" not in p.parts and ".git" not in p.parts]
    assert not bad, f"patient-level data files present: {bad}"
    assert not (APP / "all_preds").exists(), "app/all_preds must not ship"
    for outcome in appcfg.CHOSEN_MODEL_DICT:
        expl = util.load_explainer(outcome)
        if isinstance(expl, shap.TreeExplainer):
            assert expl.data is None and expl.feature_perturbation == "tree_path_dependent"
        else:
            rows = np.asarray(expl.masker.data).shape[0]
            assert rows == 1, f"{outcome} explainer stores {rows} background rows"
    print("privacy checks: no parquet/csv/xlsx; tree explainers have no background "
          "data; linear explainer background = mean row only [OK]")


def main():
    privacy_checks()
    pre = util.load_preprocessor()
    feats = util.load_reduced_features()
    cols = list(pre.feature_names_in_)
    print(f"deployed outcomes ({len(appcfg.CHOSEN_MODEL_DICT)}): {list(appcfg.CHOSEN_MODEL_DICT)}")
    print(f"raw input columns: {len(cols)}; reduced features: {len(feats)}")

    row = {c: 0 for c in cols}
    row.update(age=52, height=65, weight=240, pralbum=3.8, prwbc=7.2, prhct=39.0,
               prplate=260, prcreat=0.8, prsodm=139, asaclas=3, diabetes=1,
               fnstatus2=0, sex="female", race_new="White", acuity="elective",
               anesthes="General", surgspec="Plastics", concurrent_vhr=1)
    input_df = pd.DataFrame({c: [row[c]] for c in cols})

    ok = True
    print(f"\n{'outcome':10s} {'model':5s} {'risk':>8s} {'category':>9s} {'pctile':>7s} "
          f"{'band obs':>8s}  top-3 SHAP")
    phis = {}
    for outcome, mname in appcfg.CHOSEN_MODEL_DICT.items():
        prob, Xr = display.predict(outcome, input_df)
        cat, _, _, idx = util.get_risk_category(prob, outcome)
        thr = util.load_bin_thresholds(outcome)
        rates = util.bin_occur_rates(outcome, thr)
        pct, n_pop = util.percentile_vs_population(prob, outcome)

        expl = util.load_explainer(outcome)
        names, phi = appshap.compute_shap(expl, Xr, f"{outcome}_smoke")
        phis[outcome] = phi
        # additivity: base + sum(phi) == explained model output (log-odds margin)
        sv = expl(Xr, check_additivity=False)
        base = float(np.ravel(sv.base_values)[0])
        model = util.load_model(outcome)
        base_model = model.calibrated_classifiers_[0].estimator.estimator
        if mname == "lr":
            margin = float(base_model.decision_function(Xr)[0])
        elif mname == "xgb":
            margin = float(base_model.predict(Xr, output_margin=True)[0])
        else:
            margin = float(base_model.predict(Xr, raw_score=True)[0])
        additive = abs(base + float(np.sum(phi)) - margin) < 1e-4
        chart_spec = appshap.shap_chart(names, phi, 10).to_dict()
        additive &= bool(chart_spec.get("layer"))

        good = (0 <= prob <= 1 and np.isfinite(phi).all() and additive
                and 0 <= pct <= 100 and n_pop == 2829 and len(rates) == 4)
        ok &= good
        top = [str(names[i]) for i in np.argsort(np.abs(phi))[::-1][:3]]
        print(f"{outcome:10s} {mname:5s} {prob:8.2%} {cat:>9s} {pct:6.0f}th {rates[idx]:8.1%}  "
              f"{top}{'' if good else '   <-- FAIL'}")

    distinct = len({tuple(np.round(v, 10)) for v in phis.values()})
    print(f"\ndistinct per-outcome SHAP vectors: {distinct}/{len(phis)} "
          f"(cache-key fix: each outcome gets its own explanation)")
    ok &= distinct == len(phis)

    import sklearn, lightgbm, xgboost, streamlit
    print(f"versions: sklearn {sklearn.__version__}, lightgbm {lightgbm.__version__}, "
          f"xgboost {xgboost.__version__}, shap {shap.__version__}, numpy {np.__version__}, "
          f"pandas {pd.__version__}, streamlit {streamlit.__version__}")
    print(f"torch imported: {'torch' in sys.modules}; optuna imported: {'optuna' in sys.modules}")
    ok &= "torch" not in sys.modules and "optuna" not in sys.modules
    relevant = [w for w in caught if not issubclass(w.category, PendingDeprecationWarning)
                and "No runtime found" not in str(w.message)]
    kinds = sorted({f"{w.category.__name__}: {str(w.message)[:120]}" for w in relevant})
    print(f"warnings during load/predict: {len(relevant)}" + "".join(f"\n   - {k}" for k in kinds))
    ok &= not any("InconsistentVersion" in w.category.__name__ for w in relevant)

    print(f"\nDEPLOY SMOKE TEST: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
