"""Preprocessing pipeline for PRO-PANN.

Reuses ML-Tongue-Pred's design end-to-end -- `BMICalculatorArray`, a ColumnTransformer
combining IterativeImputer (MICE-style) for continuous vars, ordinal encoding,
one-hot for nominals, passthrough for binaries, MinMaxScaler -- with two
adaptations for this cohort:

  * the train/val/test split is a **strict temporal partition by cohort_year**
    (train<=2021 / val 2022-23 / test 2024), never reshuffled (SPEC sec. 1);
  * ordinal encoding is generalised from the framework's ASA-only handler to the
    three panniculectomy ordinals (asaclas / diabetes / fnstatus2) via a
    round-and-clip transformer.

The entire pipeline is **fit on TRAINING data only**, then used to transform val
and test (leakage guard -- acceptance check 1).
"""
from __future__ import annotations

import warnings
from shutil import rmtree

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from src import config
from src.config import SEED


# ---------------------------------------------------------------------------
#  transformers (adapted from ML-Tongue-Pred src/preprocess.py)
# ---------------------------------------------------------------------------
def remove_prefix(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X.columns = X.columns.str.replace(r"^\w+__", "", regex=True)
    return X


class BMICalculatorArray(BaseEstimator, TransformerMixin):
    """Compute imperial BMI from height+weight, drop both, append BMI last."""

    def __init__(self, height_idx, weight_idx):
        self.height_idx = height_idx
        self.weight_idx = weight_idx

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        height = X[:, self.height_idx]
        weight = X[:, self.weight_idx]
        bmi = (weight * 703) / (height ** 2)
        mask = np.ones(X.shape[1], dtype=bool)
        mask[[self.height_idx, self.weight_idx]] = False
        X_new = X[:, mask]
        X_new = np.column_stack([X_new, bmi])
        return X_new.astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        input_features = list(input_features)
        features = [f for i, f in enumerate(input_features)
                    if i not in (self.height_idx, self.weight_idx)]
        features.append("BMI")
        return np.array(features)


class RoundClipOrdinal(BaseEstimator, TransformerMixin):
    """Round to nearest int and clip each column to its (min, max) range.

    Generalises the framework's ASA rounding/clipping to multiple ordinals.
    """

    def __init__(self, ranges):
        self.ranges = ranges  # list of (lo, hi) per column

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.rint(np.asarray(X, dtype=float))
        for j, (lo, hi) in enumerate(self.ranges):
            X[:, j] = np.clip(X[:, j], lo, hi)
        return X.astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


# ---------------------------------------------------------------------------
#  preprocessor construction
# ---------------------------------------------------------------------------
def get_feature_groups(feature_cols):
    """Split the retained feature list into the four pipeline groups."""
    num = [c for c in config.NUMERIC_COLS if c in feature_cols]
    ordi = [c for c in config.ORDINAL_COLS if c in feature_cols]
    nom = [c for c in config.NOMINAL_COLS if c in feature_cols]
    binr = [c for c in config.BINARY_COLS if c in feature_cols]
    return num, ordi, nom, binr


def build_preprocessor(feature_cols):
    """Build the (unfitted) ColumnTransformer for the given feature list."""
    num, ordi, nom, binr = get_feature_groups(feature_cols)

    height_idx = num.index("height")
    weight_idx = num.index("weight")

    num_pipeline = Pipeline([
        ("imputer", IterativeImputer(
            estimator=None, initial_strategy="median",
            max_iter=10, sample_posterior=False, random_state=SEED)),
        ("bmi", BMICalculatorArray(height_idx=height_idx, weight_idx=weight_idx)),
        ("scaler", MinMaxScaler()),
    ])

    ord_ranges = [config.ORDINAL_RANGES[c] for c in ordi]
    ord_pipeline = Pipeline([
        ("imputer", IterativeImputer(
            estimator=None, initial_strategy="median",
            max_iter=10, sample_posterior=False, random_state=SEED)),
        ("round_clip", RoundClipOrdinal(ranges=ord_ranges)),
    ])

    nom_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num),
        ("ord", ord_pipeline, ordi),
        ("cat", nom_pipeline, nom),
        ("bin", "passthrough", binr),
    ])
    return preprocessor


# ---------------------------------------------------------------------------
#  temporal split + transform + export
# ---------------------------------------------------------------------------
def temporal_split(df):
    """Strict partition by cohort_year (SPEC sec. 1). No reshuffling."""
    train = df[df.cohort_year.isin(config.TRAIN_YEARS)].copy()
    val = df[df.cohort_year.isin(config.VAL_YEARS)].copy()
    test = df[df.cohort_year.isin(config.TEST_YEARS)].copy()
    assert len(train) + len(val) + len(test) == len(df)
    return train, val, test


def preprocess_once(df, feature_cols, outcome_keys, data_path, pipeline_path,
                    verbose=True):
    """Fit the preprocessor ONCE (X is identical across outcomes for a temporal
    split) and export shared X (train/val/test) + per-outcome y + the pipeline.

    Returns the fitted preprocessor and the encoded feature-name list.
    """
    train, val, test = temporal_split(df)
    preprocessor = build_preprocessor(feature_cols)
    preprocessor.fit(train[feature_cols])
    feature_names = list(preprocessor.get_feature_names_out())

    def _tx(part):
        Xt = np.array(preprocessor.transform(part[feature_cols]))
        Xt = remove_prefix(pd.DataFrame(Xt, columns=preprocessor.get_feature_names_out()))
        Xt = Xt.reset_index(drop=True)
        for c in Xt.columns:
            Xt[c] = pd.to_numeric(Xt[c], errors="coerce")
        return Xt

    X_train, X_val, X_test = _tx(train), _tx(val), _tx(test)
    data_path.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(data_path / "X_train.parquet")
    X_val.to_parquet(data_path / "X_val.parquet")
    X_test.to_parquet(data_path / "X_test.parquet")
    test["caseid"].reset_index(drop=True).to_frame("caseid").to_excel(
        data_path / "test_ids.xlsx")

    for key in outcome_keys:
        od = data_path / key
        od.mkdir(parents=True, exist_ok=True)
        train[key].reset_index(drop=True).to_frame(key).to_excel(od / "y_train.xlsx")
        val[key].reset_index(drop=True).to_frame(key).to_excel(od / "y_val.xlsx")
        test[key].reset_index(drop=True).to_frame(key).to_excel(od / "y_test.xlsx")

    pipeline_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, pipeline_path / "pipeline.joblib", compress=3)

    if verbose:
        print(f"   preprocessed once: X_train{X_train.shape} X_val{X_val.shape} "
              f"X_test{X_test.shape}; {len(feature_names)} encoded features")
    return preprocessor, list(X_train.columns)


def transform_export_data(df, feature_cols, target_col, preprocessor,
                          data_path=None, pipeline_path=None, verbose=True):
    """Temporal-split, fit-on-train, transform all splits, export.

    Returns a dict with X_train/y_train/X_val/y_val/X_test/y_test (+ test_ids).
    The preprocessor is fit on TRAIN ONLY.
    """
    train, val, test = temporal_split(df)
    X_train, y_train = train[feature_cols].copy(), train[target_col].copy()
    X_val, y_val = val[feature_cols].copy(), val[target_col].copy()
    X_test, y_test = test[feature_cols].copy(), test[target_col].copy()
    test_ids = test["caseid"].copy()

    preprocessor.fit(X_train)
    feature_names = preprocessor.get_feature_names_out()

    def _tx(X):
        Xt = np.array(preprocessor.transform(X))
        Xt = pd.DataFrame(Xt, columns=feature_names)
        return remove_prefix(Xt)

    X_train_t, X_val_t, X_test_t = _tx(X_train), _tx(X_val), _tx(X_test)
    for d in (X_train_t, X_val_t, X_test_t):
        d.reset_index(drop=True, inplace=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    test_ids = test_ids.reset_index(drop=True)

    for d in (X_train_t, X_val_t, X_test_t):
        for col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    if data_path is not None:
        outdir = data_path / target_col
        if outdir.exists():
            warnings.warn(f"Over-writing processed data at: {outdir}")
            rmtree(outdir)
        outdir.mkdir(parents=True, exist_ok=False)
        X_train_t.to_parquet(outdir / "X_train.parquet")
        y_train.to_frame(target_col).to_excel(outdir / "y_train.xlsx")
        X_val_t.to_parquet(outdir / "X_val.parquet")
        y_val.to_frame(target_col).to_excel(outdir / "y_val.xlsx")
        X_test_t.to_parquet(outdir / "X_test.parquet")
        y_test.to_frame(target_col).to_excel(outdir / "y_test.xlsx")
        test_ids.to_frame("caseid").to_excel(outdir / "test_ids.xlsx")

    if pipeline_path is not None:
        pipeline_path.mkdir(parents=True, exist_ok=True)
        p = pipeline_path / f"{target_col}_pipeline.joblib"
        if p.exists():
            p.unlink()
        joblib.dump(preprocessor, p, compress=3)

    if verbose:
        print(f"   [{target_col}] train={len(y_train)} (ev {y_train.mean():.3%}) | "
              f"val={len(y_val)} (ev {y_val.mean():.3%}) | "
              f"test={len(y_test)} (ev {y_test.mean():.3%}) | "
              f"n_feat={X_train_t.shape[1]}")

    return {
        "X_train": X_train_t, "y_train": y_train,
        "X_val": X_val_t, "y_val": y_val,
        "X_test": X_test_t, "y_test": y_test,
        "test_ids": test_ids,
    }
