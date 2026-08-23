"""
Optional feature selection utilities used by the AutoML tab.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


def select_features(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    method: str = "mutual_info",
    k: int = 50,
) -> pd.DataFrame:
    """
    Select top-k features by mutual information.
    Returns df with only selected feature columns + target.
    """
    from sklearn.feature_selection import (
        SelectKBest,
        mutual_info_classif,
        mutual_info_regression,
        f_classif,
        f_regression,
    )
    from sklearn.impute import SimpleImputer

    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    y = df[target]

    if X.empty or len(X.columns) <= k:
        return df

    # Impute for selector
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    if "regression" in str(task).lower():
        score_func = mutual_info_regression if method == "mutual_info" else f_regression
    else:
        score_func = mutual_info_classif if method == "mutual_info" else f_classif

    y_arr = y.values
    if not np.issubdtype(y_arr.dtype, np.number):
        from sklearn.preprocessing import LabelEncoder
        y_arr = LabelEncoder().fit_transform(y_arr)

    sel = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
    sel.fit(X_imp, y_arr)
    selected_cols = X.columns[sel.get_support()].tolist()

    keep = selected_cols + [target]
    other_cols = [c for c in df.columns if c not in keep and c in df.columns]
    return df[keep]