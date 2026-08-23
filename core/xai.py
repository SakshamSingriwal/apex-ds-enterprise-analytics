"""
Explainable AI – unified SHAP wrapper with permutation-importance fallback.
"""
from __future__ import annotations

import logging
from typing import Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("apex_ds.xai")


def explain_model_shap(
    model: Any,
    df: Any,
    target: Any,
    n_samples: int = 100,
) -> Dict[str, Any]:
    """
    Compute SHAP explanations.
    Falls back to permutation importance if SHAP fails.

    Returns dict: {success, summary_plot, bar_plot, error}
    """
    import pandas as pd

    df = pd.DataFrame(df) if not hasattr(df, "columns") else df
    x_df = df.drop(columns=[target])
    x_sample = x_df.sample(min(n_samples, len(x_df)), random_state=42) if len(x_df) > n_samples else x_df.copy()

    # ── Build predict wrapper ─────────────────────────────────────────────
    def _predict(x: Any) -> np.ndarray:
        if hasattr(model, "predict"):
            pred = model.predict(x)
            if hasattr(pred, "values"):
                return pred.values
            if hasattr(pred, "detach"):
                return pred.detach().numpy()
            return np.array(pred)
        raise RuntimeError("Model has no predict method")

    try:
        import shap  # type: ignore[import]

        # Choose explainer
        if hasattr(model, "get_booster"):                     # XGBoost native
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, "feature_importances_"):          # sklearn tree ensemble
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, "coef_"):                         # sklearn linear
            explainer = shap.LinearExplainer(model, x_sample)
        else:
            # KernelExplainer – works for any black-box
            bg = x_sample.sample(min(50, len(x_sample)), random_state=0)
            explainer = shap.KernelExplainer(_predict, bg)

        shap_values = explainer(x_sample)

        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, x_sample, show=False)
        fig_summary = plt.gcf()
        plt.tight_layout()

        # Bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, x_sample, plot_type="bar", show=False)
        fig_bar = plt.gcf()
        plt.tight_layout()

        return {"success": True, "summary_plot": fig_summary, "bar_plot": fig_bar, "error": ""}

    except Exception as shap_exc:
        logger.warning("SHAP failed (%s); falling back to permutation importance.", shap_exc)

        # ── Permutation-importance fallback ───────────────────────────────
        try:
            from sklearn.inspection import permutation_importance

            y_sample = df.loc[x_sample.index, target]

            perm = permutation_importance(
                model, x_sample, y_sample, n_repeats=5, random_state=42
            )
            importance = perm.importances_mean
            indices = np.argsort(importance)[::-1][:20]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(indices)), importance[indices])
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels(x_df.columns[indices])
            ax.invert_yaxis()
            ax.set_xlabel("Mean decrease in score")
            ax.set_title("Permutation Importance (SHAP unavailable)")
            plt.tight_layout()

            return {"success": True, "summary_plot": fig, "bar_plot": fig, "error": ""}

        except Exception as perm_exc:
            return {
                "success": False,
                "summary_plot": None,
                "bar_plot": None,
                "error": f"SHAP: {shap_exc}; Permutation: {perm_exc}",
            }