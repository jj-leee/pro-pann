"""SHAP helpers (adapted from ML-Tongue-Pred `src/feat_imp.py`).

`combine_encoded` sums the SHAP values of a categorical's one-hot dummies back
into a single feature so the global heatmap and per-patient explanations read in
terms of the original variables (incl. `concurrent_vhr`, the headline result).
`get_ohe_cols` is generalised to group dummies by the known nominal base names
in `config.NOMINAL_COLS` (handles multi-underscore bases like `race_new`).
"""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import shap

from src import config


def get_ohe_cols(df, nominal_bases=None):
    """Map nominal base name -> list of its one-hot dummy column suffixes."""
    nominal_bases = nominal_bases or config.NOMINAL_COLS
    groups = {}
    for col in df.columns:
        for base in nominal_bases:
            if col.startswith(base + "_"):
                groups.setdefault(base, []).append(col[len(base) + 1:])
                break
    return groups


def combine_encoded(shap_values, name, mask, return_original=True):
    """Sum SHAP values across the dummies in `mask` into one feature `name`."""
    mask = np.array(mask)
    mask_names = np.array(shap_values.feature_names, dtype="object")[mask]
    sv_name = shap.Explanation(
        shap_values.values[:, mask],
        feature_names=list(mask_names),
        data=shap_values.data[:, mask],
        base_values=shap_values.base_values,
        display_data=shap_values.display_data,
    )
    new_data = (sv_name.data * np.arange(int(mask.sum()))).sum(axis=1).astype(int)
    svdata = np.concatenate([shap_values.data[:, ~mask], new_data.reshape(-1, 1)], axis=1)
    if shap_values.display_data is None:
        svdd = shap_values.data[:, ~mask]
    else:
        svdd = shap_values.display_data[:, ~mask]
    svdisplay = np.concatenate([svdd, mask_names[new_data].reshape(-1, 1)], axis=1)

    if len(shap_values.values.shape) == 3:
        new_values = sv_name.values.sum(axis=1, keepdims=True)
        svvalues = np.concatenate([shap_values.values[:, ~mask, :], new_values], axis=1)
    else:
        new_values = sv_name.values.sum(axis=1)
        svvalues = np.concatenate(
            [shap_values.values[:, ~mask], new_values.reshape(-1, 1)], axis=1)
    svnames = list(np.array(shap_values.feature_names)[~mask]) + [name]
    sv = shap.Explanation(svvalues, base_values=shap_values.base_values, data=svdata,
                          display_data=svdisplay, feature_names=svnames)
    return (sv, sv_name) if return_original else sv


def combine_all_encoded(input_df, shap_raw):
    ohe = get_ohe_cols(input_df)
    sv = copy.deepcopy(shap_raw)
    for base in ohe:
        sv, _ = combine_encoded(sv, base, [base + "_" in n or n.startswith(base + "_")
                                           for n in sv.feature_names])
    return sv


def get_vals_to_plot(shap_vals):
    if len(shap_vals.values.shape) == 3:
        if shap_vals.values.shape[2] == 1:
            return shap_vals[:, :, 0]
        return shap_vals[:, :, 1]
    return shap_vals


def mean_abs_shap(shap_vals):
    """Mean |SHAP| per (combined) feature -> Series (for the global heatmap)."""
    sp = get_vals_to_plot(shap_vals)
    df = pd.DataFrame(sp.values, columns=sp.feature_names)
    return df.abs().mean().sort_values(ascending=False)
