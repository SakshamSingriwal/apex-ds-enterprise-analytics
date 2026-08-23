"""
Streamlit UI for the autopilot engine.

Run:  streamlit run autopilot_ui.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import autopilot as ap

st.set_page_config(page_title="DS Autopilot", layout="wide")
st.title("🚀 Data Science Autopilot")
st.caption("Upload any tabular dataset — EDA, preprocessing, imbalance handling, "
           "train/val/test, model leaderboard, SHAP, and a deployable model, all automatic.")

tab_run, tab_forecast, tab_predict, tab_history = st.tabs(
    ["Run pipeline", "Forecast", "Predict", "Past runs"])

with tab_run:
    uploaded = st.file_uploader("Dataset (CSV / XLSX / Parquet)", type=["csv", "xlsx", "parquet"])
    col1, col2 = st.columns(2)
    target_in = col1.text_input("Target column (blank = auto-detect)")
    budget = col2.number_input("Training time budget (seconds)", 60, 3600, 600, step=60)

    if uploaded and st.button("Run autopilot", type="primary"):
        tmp = ap.RUNS_DIR / "_uploads"
        tmp.mkdir(parents=True, exist_ok=True)
        data_path = tmp / uploaded.name
        data_path.write_bytes(uploaded.getbuffer())

        with st.status("Running pipeline…", expanded=True) as status:
            try:
                run_dir = ap.run(data_path, target_in or None, int(budget))
                status.update(label="Done!", state="complete")
            except SystemExit as exc:
                st.error(str(exc))
                st.stop()

        meta = json.loads((run_dir / "run_meta.json").read_text())
        st.success(f"Best model: **{meta['best_model']}** — test metrics: {meta['test_metrics']}")

        c1, c2 = st.columns(2)
        c1.subheader("Leaderboard")
        c1.dataframe(pd.read_csv(run_dir / "leaderboard.csv"))
        c2.subheader("Run metadata")
        c2.json(meta)

        shap_png = run_dir / "shap_summary.png"
        if shap_png.exists():
            st.subheader("SHAP feature importance")
            st.image(str(shap_png))

        st.subheader("EDA report")
        st.components.v1.html((run_dir / "eda_report.html").read_text(), height=500, scrolling=True)
        st.info(f"All artifacts saved to `{run_dir}`. "
                f"Serve as an API: `python autopilot.py serve \"{run_dir}\"`")

with tab_forecast:
    fc_file = st.file_uploader("Time-series dataset (CSV / XLSX / Parquet)",
                               type=["csv", "xlsx", "parquet"], key="fc")
    c1, c2, c3 = st.columns(3)
    fc_date = c1.text_input("Date column (blank = auto)")
    fc_value = c2.text_input("Value column (blank = auto)")
    fc_horizon = c3.number_input("Forecast horizon (periods)", 5, 365, 30)

    if fc_file and st.button("Run forecast", type="primary"):
        tmp = ap.RUNS_DIR / "_uploads"
        tmp.mkdir(parents=True, exist_ok=True)
        path = tmp / fc_file.name
        path.write_bytes(fc_file.getbuffer())
        with st.status("Forecasting…", expanded=True) as status:
            try:
                run_dir = ap.forecast(path, fc_date or None, fc_value or None, int(fc_horizon))
                status.update(label="Done!", state="complete")
            except SystemExit as exc:
                st.error(str(exc))
                st.stop()
        meta = json.loads((run_dir / "run_meta.json").read_text())
        st.success(f"Best model: **{meta['best_model']}** — backtest MAE: {meta['backtest_mae']}")
        png = run_dir / "forecast.png"
        if png.exists():
            st.image(str(png))
        fc_df = pd.read_csv(run_dir / "forecast.csv")
        st.dataframe(fc_df)
        st.download_button("Download forecast CSV", fc_df.to_csv(index=False),
                           file_name="forecast.csv", mime="text/csv")

with tab_predict:
    runs = sorted(ap.RUNS_DIR.glob("*/run_meta.json"), reverse=True)
    if not runs:
        st.info("No finished runs yet — run the pipeline first.")
    else:
        options = {p.parent.name: p.parent for p in runs}
        chosen = st.selectbox("Trained run", list(options))
        new_file = st.file_uploader("New data to score (CSV)", type=["csv"], key="pred")
        if new_file and st.button("Predict"):
            tmp = ap.RUNS_DIR / "_uploads"
            tmp.mkdir(parents=True, exist_ok=True)
            path = tmp / new_file.name
            path.write_bytes(new_file.getbuffer())
            out = ap.predict_file(options[chosen], path)
            preds = pd.read_csv(out)
            st.dataframe(preds.head(100))
            st.download_button("Download all predictions", preds.to_csv(index=False),
                               file_name=out.name, mime="text/csv")

with tab_history:
    runs = sorted(ap.RUNS_DIR.glob("*/run_meta.json"), reverse=True)
    if not runs:
        st.info("No runs yet.")
    for p in runs:
        meta = json.loads(p.read_text())
        with st.expander(f"{p.parent.name} — {meta.get('best_model', '?')} "
                         f"({meta.get('problem_type', meta.get('mode', 'run'))})"):
            st.json(meta)
