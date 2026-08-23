"""
Business intelligence layer: KPIs, executive summary, recommendations, risks.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np


def generate_business_insights(
    df: pd.DataFrame,
    target: Any,
    problem_type: Any,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """Generate KPIs, executive summary, recommendations and risk factors."""

    rows, cols = df.shape
    missing_pct = round(float(df.isnull().sum().sum() / (rows * cols) * 100), 2)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    kpis: Dict[str, Any] = {
        "rows": rows,
        "features": cols - (1 if target in df.columns else 0),
        "missing_%": missing_pct,
        "numeric_features": len(numeric_cols),
        "categorical_features": len(cat_cols),
    }

    # Target stats
    if target and target in df.columns:
        if pd.api.types.is_numeric_dtype(df[target]):
            kpis["target_mean"] = round(float(df[target].mean()), 4)
            kpis["target_std"] = round(float(df[target].std()), 4)
        else:
            kpis["target_classes"] = int(df[target].nunique())
            if df[target].nunique() == 2:
                vc = df[target].value_counts(normalize=True)
                kpis["minority_class_pct"] = round(float(vc.min() * 100), 2)

    # Executive summary
    ptype_label = str(problem_type).replace("_", " ").title()
    summary = (
        f"Dataset contains {rows:,} records with {kpis['features']} predictive features. "
        f"Problem type: {ptype_label}. "
        f"Data completeness: {100 - missing_pct:.1f}%. "
    )
    if model is not None:
        summary += "A trained model is available for inference and explanation. "
    summary += (
        f"Numeric features ({len(numeric_cols)}) dominate; "
        f"{len(cat_cols)} categorical columns require encoding."
    )

    # Recommendations
    recommendations: List[str] = []
    if model is not None and hasattr(model, "feature_importance"):
        try:
            fi = model.feature_importance(df)
            if fi is not None:
                col = "importance" if "importance" in (fi.columns if hasattr(fi, "columns") else []) else 0
                top3 = fi[col].head(3).index.tolist() if hasattr(fi, "columns") else []
                if top3:
                    recommendations.append(
                        f"Focus on top drivers: {', '.join(str(f) for f in top3)}. "
                        "These explain most of the model's predictive power."
                    )
        except Exception:
            pass

    if not recommendations:
        recommendations = [
            "Perform thorough EDA before modelling to identify hidden patterns.",
            "Consider feature engineering to capture domain-specific interactions.",
            "Validate model on holdout data that reflects real production distribution.",
            "Monitor model drift quarterly and retrain when performance degrades > 5%.",
        ]

    # Risk factors
    risks: List[str] = []
    if missing_pct > 20:
        risks.append(f"High missingness ({missing_pct:.1f}%) may bias model. Investigate and impute carefully.")
    if missing_pct > 5:
        risks.append("Moderate missing data detected – consider multiple imputation strategies.")

    if "minority_class_pct" in kpis and kpis["minority_class_pct"] < 20:
        risks.append(
            f"Class imbalance: minority class = {kpis['minority_class_pct']}%. "
            "Use oversampling (SMOTE) or class-weighted loss."
        )

    if rows < 1000:
        risks.append("Small dataset (<1,000 rows). Model generalisation may be limited; use cross-validation.")

    if len(cat_cols) > 20:
        risks.append("High cardinality or many categorical columns. Encoding may inflate dimensionality.")

    if not risks:
        risks.append("No major data quality risks detected. Proceed with modelling.")

    return {
        "kpis": kpis,
        "summary": summary,
        "recommendations": recommendations,
        "risks": risks,
    }