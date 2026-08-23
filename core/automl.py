"""
AutoML wrapper around AutoGluon TabularPredictor.
"""
from __future__ import annotations

import logging
import tempfile
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger("apex_ds.automl")

_PROBLEM_TYPE_MAP = {
    "binary_classification": "binary",
    "multiclass_classification": "multiclass",
    "classification": "multiclass",
    "regression": "regression",
    "ordinal_regression": "regression",
    "survival_analysis": "regression",
    "time_series_forecasting": "regression",
    "anomaly_detection": "binary",
    "multi_label_classification": "multiclass",
    "recommendation": "regression",
    "image_classification": "multiclass",
    "text_classification": "multiclass",
}


def _map_problem_type(problem_type: str) -> str:
    return _PROBLEM_TYPE_MAP.get(problem_type, "multiclass")


def train_automl(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    time_limit: int = 300,
    preset: str = "medium_quality_faster_train",
    eval_metric: Optional[str] = None,
    calibrate: bool = True,
    num_bag_folds: int = 5,
    num_stack_levels: int = 1,
):
    """Train AutoGluon predictor. Returns predictor or None on failure."""
    try:
        from autogluon.tabular import TabularPredictor  # type: ignore[import]

        ag_problem = _map_problem_type(problem_type)

        if eval_metric is None:
            if ag_problem in ("binary", "multiclass"):
                eval_metric = "accuracy"
            else:
                eval_metric = "rmse"

        save_path = os.path.join(tempfile.gettempdir(), "apex_automl_model")

        predictor = TabularPredictor(
            label=target,
            problem_type=ag_problem,
            eval_metric=eval_metric,
            path=save_path,
        ).fit(
            df,
            time_limit=time_limit,
            presets=preset,
            num_bag_folds=num_bag_folds if calibrate else 0,
            num_stack_levels=num_stack_levels if calibrate else 0,
        )
        return predictor
    except Exception as exc:
        logger.error("AutoML training failed: %s", exc)
        return None


def get_feature_importance(predictor, df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Return feature importance DataFrame sorted descending, or None."""
    try:
        fi = predictor.feature_importance(df)
        if isinstance(fi, pd.DataFrame):
            if "importance" in fi.columns:
                return fi.sort_values("importance", ascending=False)
            return fi
        # Some versions return a Series
        if isinstance(fi, pd.Series):
            return fi.sort_values(ascending=False).rename("importance").to_frame()
        return None
    except Exception as exc:
        logger.warning("Feature importance failed: %s", exc)
        return None