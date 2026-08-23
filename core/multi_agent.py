"""
Multi-agent orchestration pipeline.
"""
from __future__ import annotations

import logging
from typing import Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger("apex_ds.multi_agent")


def _explorer_agent(df: pd.DataFrame, target: Any) -> Dict[str, Any]:
    return {
        "shape": list(df.shape),
        "missing_values": int(df.isnull().sum().sum()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_features": int(len(df.select_dtypes(include=[np.number]).columns)),
        "categorical_features": int(len(df.select_dtypes(include=["object", "category"]).columns)),
        "target": str(target),
    }


def _data_quality_agent(df: pd.DataFrame) -> Dict[str, Any]:
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    duplicate_rows = int(df.duplicated().sum())
    return {
        "missing_percentage_per_column": missing_pct[missing_pct > 0].to_dict(),
        "duplicate_rows": duplicate_rows,
        "total_missing_pct": round(float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100), 2),
    }


def _model_agent(df: pd.DataFrame, target: Any, problem_type: Any) -> Dict[str, Any]:
    try:
        from core.automl import train_automl, get_feature_importance
        predictor = train_automl(df, str(target), str(problem_type), time_limit=120)
        if predictor is None:
            return {"model": "AutoML failed", "score": None, "feature_importance": {}}
        lb = predictor.leaderboard(silent=True)
        best_model = str(lb.iloc[0]["model"]) if len(lb) > 0 else "Unknown"
        best_score = float(lb.iloc[0]["score_val"]) if len(lb) > 0 else 0.0
        fi = get_feature_importance(predictor, df)
        top_feats: Dict[str, float] = {}
        if fi is not None:
            col = "importance" if "importance" in fi.columns else fi.columns[0]
            top_feats = {str(k): float(v) for k, v in fi[col].head(5).items()}
        return {"model": best_model, "score": best_score, "feature_importance": top_feats, "_predictor": predictor}
    except Exception as exc:
        logger.error("Model agent error: %s", exc)
        return {"model": "Error", "score": None, "feature_importance": {}}


def _critic_agent(score: float | None, problem_type: str) -> Dict[str, Any]:
    if score is None:
        return {"verdict": "UNKNOWN", "reason": "No score available."}
    if "regression" in problem_type.lower():
        verdict = "OK" if score < 1.0 else "UNDERFIT"
    else:
        if score >= 0.95:
            verdict = "OVERFIT"
        elif score >= 0.6:
            verdict = "OK"
        else:
            verdict = "UNDERFIT"
    return {"verdict": verdict, "score": score}


def _business_agent(top_features: Dict[str, float]) -> Dict[str, Any]:
    if not top_features:
        return {"recommendation": "Collect more data and ensure feature quality before deployment."}
    feat_names = list(top_features.keys())[:3]
    rec = (
        f"Key business drivers identified: {', '.join(feat_names)}. "
        f"Focus stakeholder attention on these variables for maximum impact. "
        f"Consider feature engineering around '{feat_names[0]}' to improve model performance."
    )
    return {"recommendation": rec, "top_features": feat_names}


def run_multi_agent_pipeline(
    df: pd.DataFrame,
    target: Any,
    problem_type: Any,
) -> Dict[str, Any]:
    """Run all agents sequentially and return combined results."""
    results: Dict[str, Any] = {}

    results["Explorer"] = _explorer_agent(df, target)
    results["Data Quality"] = _data_quality_agent(df)

    model_out = _model_agent(df, target, problem_type)
    # Don't expose the raw predictor object to the UI
    predictor_obj = model_out.pop("_predictor", None)
    results["Model"] = model_out

    score = model_out.get("score")
    results["Critic"] = _critic_agent(score, str(problem_type))
    results["Business"] = _business_agent(model_out.get("feature_importance", {}))

    return results