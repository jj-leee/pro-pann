"""Prediction/display parity between two copies of the PRO-PANN app.

Imports the app modules from --app-dir (exactly as Streamlit would), runs a
fixed set of synthetic patients through predict -> risk category -> observed
bin rate -> percentile -> SHAP for every deployed outcome, and dumps the results
(plus the exact UI strings) to JSON. Run once per app copy / environment, then
`compare`.

    python tools/parity_check.py dump --app-dir <orig>/app --out /tmp/orig.json --emit-probes
    python tools/parity_check.py dump --app-dir app --out /tmp/deploy.json --probes /tmp/orig.json
    python tools/parity_check.py compare /tmp/orig.json /tmp/deploy.json

`--emit-probes` (original copy only) also evaluates the percentile function at
every 2024-test prediction and its float neighbours (worst case for ties). Those
probes are patient-derived: keep the output JSON OUT of this repository.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore", message=".*No runtime found.*")
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

import numpy as np


def _import_app(app_dir: Path):
    sys.path.insert(0, str(app_dir))
    import logging
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    import config as appcfg  # noqa: F401  (adds repo root to sys.path)
    import utils as util
    import display_functions as display
    import shap_utils as appshap
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("streamlit"):
            logging.getLogger(name).setLevel(logging.ERROR)
    return appcfg, util, display, appshap


def synthetic_patients(util, cols, n_random=40, seed=20241):
    rng = np.random.default_rng(seed)
    base = {"age": 52, "height": 65, "weight": 240, "pralbum": 3.8, "prwbc": 7.2,
            "prhct": 39.0, "prplate": 260, "prcreat": 0.8, "prsodm": 139,
            "asaclas": 3, "diabetes": 1, "fnstatus2": 0, "sex": "female",
            "race_new": "White", "acuity": "elective", "anesthes": "General",
            "surgspec": "Plastics", "concurrent_vhr": 1}

    def full(d):
        return {c: d.get(c, 0) for c in cols}

    pats = [full(base), full({**base, "concurrent_vhr": 0}),
            full({**base, "pralbum": np.nan, "prcreat": np.nan, "prsodm": np.nan,
                  "prwbc": np.nan, "prhct": np.nan, "prplate": np.nan}),
            full({**base, "age": 78, "weight": 420, "asaclas": 4, "diabetes": 2,
                  "smoke": 1, "steroid": 1, "hxcopd": 1, "fnstatus2": 2,
                  "abdominoplasty": 1, "concurrent_liposuction": 1, "acuity": "urgent"})]
    nums = {"age": (18, 90), "height": (55, 76), "weight": (110, 500),
            "pralbum": (2.0, 5.0), "prwbc": (3, 18), "prhct": (25, 50),
            "prplate": (100, 500), "prcreat": (0.4, 3.0), "prsodm": (128, 146)}
    cats = {"sex": util.SEX_OPTIONS, "race_new": util.RACE_OPTIONS,
            "acuity": util.ACUITY_OPTIONS, "anesthes": util.ANESTH_OPTIONS,
            "surgspec": util.SURGSPEC_OPTIONS}
    ords = {"asaclas": [1, 2, 3, 4], "diabetes": [0, 1, 2], "fnstatus2": [0, 1, 2]}
    for _ in range(n_random):
        d = {}
        for c in cols:
            if c in nums:
                lo, hi = nums[c]
                d[c] = np.nan if (c.startswith("pr") and rng.random() < 0.2) else round(float(rng.uniform(lo, hi)), 1)
            elif c in cats:
                d[c] = cats[c][rng.integers(len(cats[c]))]
            elif c in ords:
                d[c] = ords[c][rng.integers(len(ords[c]))]
            else:
                d[c] = int(rng.random() < 0.15)
        pats.append(d)
    return pats


def dump(args):
    import pandas as pd
    appcfg, util, display, appshap = _import_app(args.app_dir.resolve())
    pre = util.load_preprocessor()
    cols = list(pre.feature_names_in_)
    pats = synthetic_patients(util, cols)
    out = {"patients": [], "probes": {}}
    probes_in = json.load(open(args.probes))["probes"] if args.probes else None

    for i, p in enumerate(pats):
        input_df = pd.DataFrame({c: [p[c]] for c in cols})
        rec = {}
        for outcome in appcfg.CHOSEN_MODEL_DICT:
            prob, Xr = display.predict(outcome, input_df)
            cat, _, _, idx = util.get_risk_category(prob, outcome)
            thr = util.load_bin_thresholds(outcome)
            rates = util.bin_occur_rates(outcome, thr)
            pct, n_pop = util.percentile_vs_population(prob, outcome)
            expl = util.load_explainer(outcome)
            appshap.compute_shap.clear()  # compare true per-outcome SHAP (bypass cache)
            names, phi = appshap.compute_shap(expl, Xr, f"{outcome}_{i}")
            rec[outcome] = {
                "prob": float(prob),
                "ui": [f"{prob:.2%}", cat, f"{pct:.0f}th", f"{rates[idx]:.1%}", f"{n_pop:,}"],
                "shap_names": list(names), "shap_phi": [float(v) for v in phi],
            }
        out["patients"].append(rec)

    for outcome in appcfg.CHOSEN_MODEL_DICT:
        if args.emit_probes:
            model = appcfg.CHOSEN_MODEL_DICT[outcome]
            s = pd.read_parquet(appcfg.ALLPREDS_DIR / f"{outcome}_{model}.parquet")["prob"].to_numpy()
            grid = np.concatenate([s, np.nextafter(s, 0), np.nextafter(s, 1),
                                   np.linspace(0, 1, 2001), [s.min() / 2, 1.0]])
            probes = [float(x) for x in np.unique(grid)]
        elif probes_in is not None:
            probes = probes_in[outcome]["p"]
        else:
            probes = [float(x) for x in np.linspace(0, 1, 2001)]
        out["probes"][outcome] = {
            "p": probes,
            "pct": [f"{util.percentile_vs_population(x, outcome)[0]:.0f}" for x in probes]}
    json.dump(out, open(args.out, "w"))
    print(f"wrote {args.out}: {len(pats)} patients x {len(appcfg.CHOSEN_MODEL_DICT)} outcomes")


def compare(args):
    a, b = json.load(open(args.a)), json.load(open(args.b))
    max_dp, max_dphi, ui_mismatch, name_mismatch, n = 0.0, 0.0, [], 0, 0
    for i, (ra, rb) in enumerate(zip(a["patients"], b["patients"])):
        for o in ra:
            n += 1
            max_dp = max(max_dp, abs(ra[o]["prob"] - rb[o]["prob"]))
            if ra[o]["shap_names"] != rb[o]["shap_names"]:
                name_mismatch += 1
            else:
                max_dphi = max(max_dphi, float(np.max(np.abs(
                    np.subtract(ra[o]["shap_phi"], rb[o]["shap_phi"])))))
            if ra[o]["ui"] != rb[o]["ui"]:
                ui_mismatch.append((i, o, ra[o]["ui"], rb[o]["ui"]))
    pct_bad, pct_n = 0, 0
    for o in a["probes"]:
        pa, pb = a["probes"][o], b["probes"][o]
        assert pa["p"] == pb["p"], f"probe sets differ for {o}"
        pct_n += len(pa["pct"])
        pct_bad += sum(x != y for x, y in zip(pa["pct"], pb["pct"]))
    print(f"patient x outcome pairs compared : {n}")
    print(f"max |delta prob|                 : {max_dp:.3g}")
    print(f"max |delta SHAP phi|             : {max_dphi:.3g}  (feature-name mismatches: {name_mismatch})")
    print(f"UI-string mismatches             : {len(ui_mismatch)}")
    for m in ui_mismatch[:10]:
        print("   ", m)
    print(f"percentile probes compared       : {pct_n}  mismatches: {pct_bad}")
    ok = max_dp == 0 and max_dphi < 1e-12 and not ui_mismatch and not name_mismatch and pct_bad == 0
    print(f"PARITY: {'PASS (identical)' if ok else 'DIFFERENCES FOUND'}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--app-dir", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--probes", type=Path)
    d.add_argument("--emit-probes", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    a = ap.parse_args()
    sys.exit(dump(a) if a.cmd == "dump" else compare(a))
