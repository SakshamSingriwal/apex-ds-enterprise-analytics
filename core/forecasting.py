"""
Time-series forecasting with Facebook Prophet.
"""
from __future__ import annotations

import logging
from typing import Dict, Any

import pandas as pd

logger = logging.getLogger("apex_ds.forecasting")


def run_forecast(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    horizon: int = 30,
) -> Dict[str, Any]:
    """
    Run Prophet forecast. Returns dict with keys:
        success (bool), forecast (DataFrame), plot (plotly Figure), error (str).
    """
    try:
        from prophet import Prophet  # type: ignore[import]
        import plotly.graph_objects as go

        # Build prophet df
        prophet_df = df[[date_col, value_col]].copy()
        prophet_df.columns = ["ds", "y"]

        # Coerce date column
        try:
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        except Exception as e:
            return {"success": False, "error": f"Cannot parse date column '{date_col}': {e}"}

        prophet_df = prophet_df.dropna()
        prophet_df["y"] = pd.to_numeric(prophet_df["y"], errors="coerce")
        prophet_df = prophet_df.dropna()

        if len(prophet_df) < 5:
            return {"success": False, "error": "Need at least 5 non-null rows for forecasting."}

        model = Prophet(yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality="auto")
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)

        # Build Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"], name="Actual", mode="markers+lines"))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast", line=dict(color="#58a6ff")))
        fig.add_trace(go.Scatter(
            x=list(forecast["ds"]) + list(forecast["ds"][::-1]),
            y=list(forecast["yhat_upper"]) + list(forecast["yhat_lower"][::-1]),
            fill="toself",
            fillcolor="rgba(88,166,255,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Band",
        ))
        fig.update_layout(
            title=f"Prophet Forecast — {value_col}",
            xaxis_title="Date",
            yaxis_title=value_col,
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
        )

        return {"success": True, "forecast": forecast, "plot": fig, "error": ""}

    except ImportError:
        return {"success": False, "error": "Prophet not installed. Run: pip install prophet"}
    except Exception as exc:
        logger.exception("Forecasting error")
        return {"success": False, "error": str(exc)}