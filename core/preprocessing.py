"""
Preprocessing utilities.
NOTE: Full preprocessing (scaling/encoding) is applied AFTER train/test split
inside training functions to avoid target leakage. This module exposes a
`preview_clean` helper for UI display only.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Optional


def preview_clean(df: pd.DataFrame, target: Optional[str] = None) -> pd.DataFrame:
    """Return a cleaned copy of df for display – no target leakage."""
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target:
            continue
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col == target:
            continue
        mode = df[col].mode()
        df[col] = df[col].fillna(mode[0] if len(mode) else "MISSING")
    return df


def build_preprocessor(X: pd.DataFrame, cat_threshold: int = 10) -> ColumnTransformer:
    """Build a ColumnTransformer with RobustScaler + OHE/LE."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    low_cat_cols = [
        c for c in X.select_dtypes(include=["object", "category"]).columns
        if X[c].nunique() <= cat_threshold
    ]
    high_cat_cols = [
        c for c in X.select_dtypes(include=["object", "category"]).columns
        if X[c].nunique() > cat_threshold
    ]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
    ])
    low_cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    high_cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("le", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if low_cat_cols:
        transformers.append(("low_cat", low_cat_pipe, low_cat_cols))
    if high_cat_cols:
        transformers.append(("high_cat", high_cat_pipe, high_cat_cols))

    return ColumnTransformer(transformers, remainder="drop")