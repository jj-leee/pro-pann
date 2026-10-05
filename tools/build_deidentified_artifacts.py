"""Build the de-identified runtime artifacts for the public PRO-PANN app.

Run ONCE, locally, against the original (private) project's `app/` folder. It
reads patient-level files that must never be committed and writes only
aggregate summaries into this repo:

1. `app/all_preds/<outcome>_<model>.parquet` (2,829 per-patient 2024-test
   predictions + outcome labels) -> `app/population_summary/<outcome>_<model>.json`:
     * `bin_observed_rates`: observed event rate in each of the 4 risk bins
       (identical to the app's former `bin_occur_rates`);
     * `percentile_cutpoints` / `percentile_values`: a <=101-entry lookup that
       reproduces the app's displayed integer percentile
       (`f"{(probs < p).mean() * 100:.0f}"`) exactly, without labels or per-patient values;
     * `n_test`: test-cohort size shown in the UI.
2. Any `shap.LinearExplainer` whose masker stores background training rows
   (the deep_ssi LR explainer stores 100 rows) is rebuilt from the summarized
   background (mean vector + covariance). Interventional linear SHAP values are
   coef * (x - mean), so outputs are unchanged.

Usage (from the repo root, with the training-time package versions):
    python tools/build_deidentified_artifacts.py --orig "/path/to/ML-Panni-Pred/app"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app"


def percentile_lookup(probs: np.ndarray):
    """Exact lookup for the app's displayed percentile.

    The original app shows ``f"{(probs < p).mean() * 100:.0f}"``. With
    s = sorted(probs), k(p) = #{s_i < p} and the displayed integer d(k) is
    monotone in k. For each displayed value d (after the first), let k_d be the
    smallest k giving d; then k(p) >= k_d  <=>  s[k_d - 1] < p. Storing the
    cut-points s[k_d - 1] (~1 per percentile point) reproduces the display exactly.
    """
    s = np.sort(np.asarray(probs, dtype=float))
    n = len(s)
    disp = []
    for k in range(n + 1):
        frac = (np.arange(n) < k).mean() * 100  # same op chain as the app
        disp.append(int(f"{frac:.0f}"))
    values, cuts = [disp[0]], []
    for k in range(1, n + 1):
        if disp[k] != disp[k - 1]:
            values.append(disp[k])
            cuts.append(float(s[k - 1]))
    return values, cuts


def bin_rates(probs, labels, thresholds):
    idx = np.digitize(probs, thresholds, right=False)
    out = []
    for b in range(len(thresholds) + 1):
        m = idx == b
        out.append(None if m.sum() == 0 else float(labels[m].mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True, type=Path,
                    help="original (private) project's app/ directory")
    args = ap.parse_args()
    orig = args.orig.resolve()
    chosen = json.load(open(APP / "chosen_models.json"))

    out_dir = APP / "population_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    for outcome, model in chosen.items():
        df = pd.read_parquet(orig / "all_preds" / f"{outcome}_{model}.parquet")
        probs = df["prob"].to_numpy()
        labels = df["label"].to_numpy()
        thr = np.load(APP / "bin_thresholds" / f"{outcome}_{model}.npz")["thresholds"]
        values, cuts = percentile_lookup(probs)
        summary = {
            "outcome": outcome,
            "model": model,
            "source": "ACS-NSQIP 2024 held-out test cohort (aggregate only; no patient-level rows)",
            "n_test": int(len(probs)),
            "bin_thresholds": [float(t) for t in thr],
            "bin_observed_rates": bin_rates(probs, labels, thr),
            "percentile_values": values,
            "percentile_cutpoints": cuts,
        }
        json.dump(summary, open(out_dir / f"{outcome}_{model}.json", "w"), indent=1)
        print(f"{outcome:10s} n={len(probs)} rates={summary['bin_observed_rates']} "
              f"cutpoints={len(cuts)}")

    # Linear explainers: replace stored background rows with (mean, cov)
    for outcome in chosen:
        src = orig / "shap_explainers" / f"{outcome}.joblib"
        expl = joblib.load(src)
        if not isinstance(expl, shap.LinearExplainer):
            continue
        bg = np.asarray(expl.masker.data, dtype=float)
        mean = np.asarray(expl.mean, dtype=float)
        cov = np.cov(bg, rowvar=False)
        new = shap.LinearExplainer(expl.model, (mean, cov),
                                   feature_names=list(expl.feature_names))
        assert np.array_equal(new.mean, expl.mean)
        # expected_value = coef.mean + intercept; the original was computed on a
        # different BLAS (Windows) and can differ by ~1 ULP -> carry it over exactly.
        assert np.isclose(new.expected_value, expl.expected_value, rtol=0, atol=1e-12)
        new.expected_value = expl.expected_value
        assert np.asarray(new.masker.data).shape[0] == 1  # only the mean row
        joblib.dump(new, APP / "shap_explainers" / f"{outcome}.joblib")
        print(f"{outcome:10s} LinearExplainer rebuilt: background {bg.shape} -> "
              f"mean{mean.shape} + cov{cov.shape}")


if __name__ == "__main__":
    main()
