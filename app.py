"""
Apex DS – Enterprise Analytics Studio
Run: streamlit run app.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from core.automl import get_feature_importance, train_automl
from core.business_insights import generate_business_insights
from core.clustering import run_clustering
from core.deep_learning import get_available_architectures, train_deep_learning
from core.forecasting import run_forecast
from core.goal_agent import run_goal_agent
from core.multi_agent import run_multi_agent_pipeline
from core.rag import RAGPipeline
from core.reports import export_report
from core.sql_agent import SQLAgent
from core.utils import check_ollama, setup_logging
from core.xai import explain_model_shap

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Apex DS | Enterprise Analytics Studio",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = setup_logging()

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "dark_mode": True,
    "current_df": None,
    "current_target": None,
    "current_problem": None,
    "problem_type": None,
    "current_model": None,
    "current_model_type": None,
    "deep_learning_history": None,
    "rag_pipeline": None,
    "current_forecast": None,
    "data_profile": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Theme CSS ──────────────────────────────────────────────────────────────────

def _theme_css(dark: bool) -> str:
    if dark:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background: linear-gradient(180deg, #0a0c10 0%, #12161d 100%); }
        .stSidebar { background: linear-gradient(180deg, #0d1117 0%, #161b22 100%); }
        h1, h2, h3 { color: #58a6ff !important; font-weight: 600 !important; }
        .stButton>button {
            background: linear-gradient(135deg, #1f6feb, #388bfd);
            color: white; border-radius: 8px; border: none; font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover { transform: translateY(-2px); background: linear-gradient(135deg, #388bfd, #50a1ff); }
        .stMetric label { color: #8b949e !important; font-size: 14px !important; }
        .stMetric [data-testid="stMetricValue"] { color: #58a6ff !important; font-weight: 600 !important; }
        .stProgress > div > div { background-color: #39d353 !important; }
        .dataset-card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 16px; margin: 8px 0; }
        .classification-badge { background: linear-gradient(135deg, #39d353, #238636); color: #fff; display: inline-block; padding: 4px 12px; border-radius: 16px; font-size: 12px; font-weight: 600; }
        .regression-badge { background: linear-gradient(135deg, #f78166, #db4814); color: #fff; display: inline-block; padding: 4px 12px; border-radius: 16px; font-size: 12px; font-weight: 600; }
        .footer-text { text-align: center; padding: 1rem; color: #8b949e; font-size: 0.9rem; }
        </style>"""
    else:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background: #ffffff; }
        .stSidebar { background: #f6f8fa; }
        h1, h2, h3 { color: #1f6feb !important; font-weight: 600 !important; }
        .stButton>button { background: #1f6feb; color: white; border-radius: 8px; border: none; font-weight: 600; }
        .stButton>button:hover { background: #388bfd; }
        .stMetric label { color: #57606a !important; }
        .stMetric [data-testid="stMetricValue"] { color: #1f6feb !important; font-weight: 600 !important; }
        .dataset-card { background: #ffffff; border: 1px solid #d0d7de; border-radius: 12px; padding: 16px; margin: 8px 0; }
        .classification-badge { background: #dafbe1; color: #1a7f37; display: inline-block; padding: 4px 12px; border-radius: 16px; font-size: 12px; font-weight: 600; }
        .regression-badge { background: #ffebe9; color: #cf222e; display: inline-block; padding: 4px 12px; border-radius: 16px; font-size: 12px; font-weight: 600; }
        .footer-text { text-align: center; padding: 1rem; color: #57606a; font-size: 0.9rem; }
        </style>"""


st.markdown(_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _maybe_balloons() -> None:
    try:
        st.balloons()
    except Exception:
        pass


def detect_problem_type(df: pd.DataFrame, target: Optional[str]) -> str:
    if target is None or target not in df.columns:
        return "clustering"
    y = df[target]
    if pd.api.types.is_datetime64_any_dtype(y):
        return "time_series_forecasting"
    if pd.api.types.is_numeric_dtype(y):
        nu = int(y.nunique())
        if nu <= 2:
            return "binary_classification"
        if nu <= 20:
            return "multiclass_classification"
        return "regression"
    return "multiclass_classification"


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    if st.button("🌙 Dark mode" if st.session_state.dark_mode else "☀️ Light mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    st.markdown("---")
    st.header("APEX ANALYTICS")
    st.caption("Enterprise Analytics Studio")

    # ── Data source ───────────────────────────────────────────────────────
    with st.expander("📂 Data Source", expanded=True):
        demo = st.selectbox(
            "Load demo dataset",
            ["None", "Iris (Classification)", "Churn (Classification)",
             "Housing (Regression)", "Sales Time Series"],
        )
        uploaded = st.file_uploader("Upload CSV / XLSX", type=["csv", "xlsx", "xls"])

        if st.button("🎲 Surprise Me"):
            demo = random.choice([
                "Iris (Classification)", "Churn (Classification)",
                "Housing (Regression)", "Sales Time Series",
            ])
            st.rerun()

    # ── Load demo ─────────────────────────────────────────────────────────
    if demo != "None":
        if demo == "Iris (Classification)":
            from sklearn.datasets import load_iris
            _d = load_iris()
            df = pd.DataFrame(_d.data, columns=list(_d.feature_names))
            df["target"] = _d.target
            target = "target"
        elif demo == "Churn (Classification)":
            rng = np.random.default_rng(42)
            df = pd.DataFrame({
                "tenure": rng.integers(1, 72, 1000),
                "monthly_charges": rng.uniform(20, 120, 1000),
                "age": rng.integers(18, 70, 1000),
                "contract": rng.choice(["monthly", "yearly"], 1000),
                "churn": rng.binomial(1, 0.25, 1000),
            })
            target = "churn"
        elif demo == "Housing (Regression)":
            from sklearn.datasets import fetch_california_housing
            _d = fetch_california_housing()
            df = pd.DataFrame(_d.data, columns=list(_d.feature_names))
            df["target"] = _d.target
            target = "target"
        else:
            _rng = pd.date_range("2024-01-01", periods=120, freq="D")
            df = pd.DataFrame({
                "date": _rng,
                "sales": np.sin(np.linspace(0, 10, 120)) * 100 + 150 + np.random.normal(0, 5, 120),
            })
            target = "sales"

        st.session_state.current_df = df
        st.session_state.current_target = target
        st.session_state.current_problem = detect_problem_type(df, target)
        st.success(f"Loaded: {demo}")

    elif uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.current_df = df
            target = st.selectbox("Select target column", df.columns.tolist())
            st.session_state.current_target = target
            st.session_state.current_problem = detect_problem_type(df, target)
            st.success("Data loaded successfully.")
        except Exception as _err:
            st.error(f"Failed to load dataset: {_err}")

    # ── Dataset summary card ──────────────────────────────────────────────
    if st.session_state.current_df is not None:
        _df = st.session_state.current_df
        st.markdown("<div class='dataset-card'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Rows", f"{_df.shape[0]:,}")
        with c2:
            st.metric("Cols", f"{_df.shape[1]}")
        _miss = round(float(_df.isnull().sum().sum() / (_df.shape[0] * _df.shape[1]) * 100), 1)
        _ptype = (st.session_state.current_problem or "unknown").replace("_", " ").capitalize()
        _badge = "classification-badge" if "class" in (st.session_state.current_problem or "") else "regression-badge"
        st.markdown(f"<span class='{_badge}'>{_ptype}</span>", unsafe_allow_html=True)
        st.metric("Missing %", f"{_miss:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Problem type selector ─────────────────────────────────────────────
    PROBLEM_TYPES = [
        "Binary Classification", "Multiclass Classification", "Regression",
        "Time Series Forecasting", "Clustering", "Anomaly Detection",
        "Multi-Label Classification", "Ordinal Regression", "Survival Analysis",
        "Recommendation (Implicit)", "Image Classification", "Text Classification",
    ]
    _PT_MAP = {
        "Binary Classification": "binary_classification",
        "Multiclass Classification": "multiclass_classification",
        "Regression": "regression",
        "Time Series Forecasting": "time_series_forecasting",
        "Clustering": "clustering",
        "Anomaly Detection": "anomaly_detection",
        "Multi-Label Classification": "multi_label_classification",
        "Ordinal Regression": "ordinal_regression",
        "Survival Analysis": "survival_analysis",
        "Recommendation (Implicit)": "recommendation",
        "Image Classification": "image_classification",
        "Text Classification": "text_classification",
    }

    with st.expander("🧭 Problem Type", expanded=True):
        _detected = (st.session_state.current_problem or "multiclass_classification").replace("_", " ").title()
        _default_label = _detected if _detected in PROBLEM_TYPES else PROBLEM_TYPES[1]
        _selected_label = st.selectbox(
            "Problem type (auto-detected)",
            PROBLEM_TYPES,
            index=PROBLEM_TYPES.index(_default_label),
        )
        st.session_state.problem_type = _PT_MAP[_selected_label]

    show_advanced = st.checkbox("Advanced mode (all tabs)", value=False)

    st.markdown("---")
    st.caption("⚡ Local-first · No cloud · Free forever")

# ── Main header ────────────────────────────────────────────────────────────────
st.title("◆ Apex DS")
st.caption("Enterprise Analytics Studio · AutoML · Deep Learning · RAG · Multi-Agent")

if st.session_state.current_df is None:
    st.info("👈 Load a dataset from the sidebar to begin.")
    st.stop()

# ── Dynamic tab selection ──────────────────────────────────────────────────────
ALL_TABS = [
    "🤖 AutoML",
    "📊 Data Profiling",
    "🧠 Deep Learning",
    "📊 Clustering",
    "📈 Forecasting",
    "💬 RAG Chat",
    "🗄️ SQL Agent",
    "🔍 Explainable AI",
    "🤝 Multi-Agent",
    "🎯 Goal Agent",
    "📊 Business Analytics",
    "📄 Reports",
    "🎮 RL",
]

ptype = st.session_state.problem_type or ""

if ptype == "time_series_forecasting":
    relevant_tabs = ["📈 Forecasting", "📊 Data Profiling", "📄 Reports", "🤝 Multi-Agent", "🎯 Goal Agent"]
elif ptype in ("clustering", "anomaly_detection"):
    relevant_tabs = ["📊 Clustering", "📊 Data Profiling", "📄 Reports"]
elif ptype in (
    "binary_classification", "multiclass_classification", "multi_label_classification",
    "ordinal_regression", "survival_analysis", "regression", "recommendation",
    "image_classification", "text_classification",
):
    relevant_tabs = [t for t in ALL_TABS if t != "🎮 RL"]
else:
    relevant_tabs = ALL_TABS

tab_names = ALL_TABS if show_advanced else relevant_tabs
tabs = st.tabs(tab_names)
tab_idx = {name: i for i, name in enumerate(tab_names)}

# ── Convenience aliases ────────────────────────────────────────────────────────
_ss = st.session_state

# ══════════════════════════════════════════════════════════════════════════════
# TAB: AutoML
# ══════════════════════════════════════════════════════════════════════════════
if "🤖 AutoML" in tab_idx:
    with tabs[tab_idx["🤖 AutoML"]]:
        st.header("🤖 AutoML Engine")
        st.markdown("Automatically trains and compares multiple models using AutoGluon.")

        # Keyboard shortcut hint
        st.markdown("""
        <script>
        document.addEventListener('keydown', function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                const btns = window.parent.document.querySelectorAll('button');
                for (const b of btns) { if (b.innerText.trim().startsWith('🚀 Run AutoML')) { b.click(); break; } }
            }
        });
        </script>""", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            time_limit = st.slider("Training time limit (s)", 60, 600, 300)
            auto_fs = st.checkbox("Auto feature selection (Mutual Info)", value=True)
            run_btn = st.button("🚀 Run AutoML", type="primary", help="Ctrl+Enter")

        if run_btn:
            _maybe_balloons()
            predictor = None
            with st.spinner("Training models… this may take a few minutes."):
                try:
                    df_train = _ss.current_df.copy()
                    t_col = _ss.current_target

                    if auto_fs and t_col and t_col in df_train.columns:
                        try:
                            from core.feature_selection import select_features
                            num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
                            k = min(50, len(num_cols))
                            if k > 0:
                                df_train = select_features(df_train, t_col, task=_ss.current_problem or "classification", k=k)
                        except Exception:
                            pass

                    # Progress bar simulation
                    prog = st.progress(0, text="Initialising AutoGluon…")
                    for _pct in range(0, 80, 20):
                        import time
                        time.sleep(0.1)
                        prog.progress(_pct, text=f"Training models… {_pct}%")

                    predictor = train_automl(
                        df_train,
                        str(t_col),
                        str(_ss.current_problem),
                        time_limit=time_limit,
                    )
                    prog.progress(100, text="Done!")

                    if predictor is not None:
                        _ss.current_model = predictor
                        _ss.current_model_type = "automl"
                        st.success("✅ Training complete!")
                    else:
                        st.error("AutoML returned None. Check logs.")

                except Exception as exc:
                    st.error(f"AutoML failed: {exc}")

            # ── Results (inside the if block so predictor is bound) ──────
            if predictor is not None:
                try:
                    leaderboard = predictor.leaderboard(silent=True)
                except Exception:
                    leaderboard = None

                if leaderboard is not None and len(leaderboard) > 0:
                    st.subheader("🏆 Model Leaderboard")
                    st.dataframe(leaderboard[["model", "score_val", "fit_time", "pred_time_val"]].head(10))
                    best_row = leaderboard.iloc[0]
                    st.metric("Best Model", str(best_row["model"]), f"Score: {float(best_row['score_val']):.4f}")

                    st.subheader("📊 Feature Importance")
                    with st.spinner("Computing feature importance…"):
                        fi = get_feature_importance(predictor, _ss.current_df)

                    if fi is not None:
                        fi_col = "importance" if "importance" in fi.columns else fi.columns[0]
                        fig_fi = px.bar(
                            fi.head(20), x=fi_col, y=fi.head(20).index,
                            orientation="h", title="Top 20 Feature Importances",
                        )
                        fig_fi.update_layout(
                            template="plotly_dark", paper_bgcolor="#0d1117",
                            plot_bgcolor="#0d1117", yaxis={"autorange": "reversed"},
                        )
                        st.plotly_chart(fig_fi, use_container_width=True)

                        # Executive Insight button
                        if st.button("🧠 Executive Insight (Ollama)"):
                            if check_ollama():
                                try:
                                    import ollama  # type: ignore[import]
                                    top_feats = ", ".join(fi.head(3).index.tolist())
                                    prompt = (
                                        f"The best ML model is '{best_row['model']}' with score "
                                        f"{float(best_row['score_val']):.4f}. "
                                        f"Top 3 features: {top_feats}. "
                                        "Write a 2-sentence executive summary and one business recommendation."
                                    )
                                    resp = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
                                    insight = resp.get("message", {}).get("content", str(resp))
                                    st.info(f"💡 {insight}")
                                except Exception as ollamaerr:
                                    st.warning(f"Ollama error: {ollamaerr}")
                            else:
                                st.warning("Ollama not running. Start with `ollama serve` and pull mistral.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: Data Profiling
# ══════════════════════════════════════════════════════════════════════════════
with tabs[tab_idx["📊 Data Profiling"]]:
    st.header("📊 Data Profiling")
    df_p = _ss.current_df
    target_p = _ss.current_target

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows", f"{df_p.shape[0]:,}")
    with c2:
        st.metric("Columns", df_p.shape[1])
    with c3:
        st.metric("Numeric", len(df_p.select_dtypes(include=[np.number]).columns))
    with c4:
        st.metric("Categorical", len(df_p.select_dtypes(include=["object", "category"]).columns))

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    num_df = df_p.select_dtypes(include=[np.number])
    if len(num_df.columns) >= 2:
        corr = num_df.corr()
        fig_corr = px.imshow(
            corr, color_continuous_scale="RdBu", aspect="auto",
            labels={"color": "Correlation"},
        )
        fig_corr.update_layout(height=500, template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

    # Missing values
    st.subheader("Missing Values")
    miss = df_p.isnull().sum()
    if miss.sum() > 0:
        miss_df = pd.DataFrame({"Column": miss.index, "Missing": miss.values,
                                 "Pct": (miss / len(df_p) * 100).round(2).values})
        miss_df = miss_df[miss_df["Missing"] > 0].sort_values("Missing", ascending=False)
        fig_miss = px.bar(miss_df.head(20), x="Missing", y="Column", orientation="h",
                          color="Pct", color_continuous_scale="reds")
        fig_miss.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
        st.plotly_chart(fig_miss, use_container_width=True)
        st.dataframe(miss_df)
    else:
        st.success("✅ No missing values detected!")

    # Distribution
    st.subheader("Feature Distribution")
    dist_cols = df_p.select_dtypes(include=[np.number]).columns.tolist()
    if dist_cols:
        sel_col = st.selectbox("Feature", dist_cols)
        fig_dist = px.histogram(df_p, x=sel_col, nbins=50, marginal="box",
                                title=f"Distribution of {sel_col}")
        fig_dist.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
        st.plotly_chart(fig_dist, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB: Deep Learning
# ══════════════════════════════════════════════════════════════════════════════
if "🧠 Deep Learning" in tab_idx:
    with tabs[tab_idx["🧠 Deep Learning"]]:
        st.header("🧠 Deep Learning Lab")
        st.markdown("Train custom neural networks: ANN, 1D CNN, LSTM, GRU, TabTransformer.")

        col1, col2 = st.columns(2)
        with col1:
            arch = st.selectbox("Architecture", get_available_architectures())
            epochs = st.slider("Epochs", 10, 200, 80)
            batch_size = st.slider("Batch size", 16, 128, 32)
            lr = st.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3)
        with col2:
            dropout = st.slider("Dropout", 0.0, 0.7, 0.3, 0.05)
            patience = st.slider("Early stopping patience", 3, 30, 12)
            tune_hp = st.checkbox("Auto-tune hyperparameters (Optuna)", value=False)

        if st.button("🧠 Train Model", type="primary"):
            _maybe_balloons()
            with st.spinner("Training neural network…"):
                try:
                    model, history = train_deep_learning(
                        _ss.current_df,
                        _ss.current_target,
                        _ss.current_problem,
                        architecture=arch,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=float(lr),
                        patience=patience,
                        tune_hyperparameters=tune_hp,
                        dropout=dropout,
                    )
                    _ss.current_model = model
                    _ss.current_model_type = "deep_learning"
                    _ss.deep_learning_history = history
                    st.success("✅ Training complete!")
                except Exception as exc:
                    st.error(f"Deep learning error: {exc}")
                    history = None

            if _ss.deep_learning_history is not None:
                h = _ss.deep_learning_history
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scatter(y=h["train_loss"], name="Train Loss", line=dict(color="#58a6ff")))
                fig_lc.add_trace(go.Scatter(y=h["val_loss"], name="Val Loss", line=dict(color="#3fb950")))
                fig_lc.update_layout(title="Learning Curves", xaxis_title="Epoch", yaxis_title="Loss",
                                     template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
                st.plotly_chart(fig_lc, use_container_width=True)

                gap = abs(h["train_loss"][-1] - h["val_loss"][-1])
                if gap > 0.05:
                    st.warning(f"⚠ Possible overfitting (val gap = {gap:.3f}). Consider more dropout or fewer epochs.")
                else:
                    st.success(f"✅ Well-fitted (val gap = {gap:.3f})")

                st.markdown(f"**Architecture:** {h.get('architecture', arch)} | "
                            f"**Input dim:** {h.get('input_dim')} | "
                            f"**Output dim:** {h.get('output_dim')} | "
                            f"**Task:** {h.get('problem_type')}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: Clustering
# ══════════════════════════════════════════════════════════════════════════════
if "📊 Clustering" in tab_idx:
    with tabs[tab_idx["📊 Clustering"]]:
        st.header("📊 Clustering Studio")
        method = st.selectbox("Method", ["kmeans", "dbscan", "gaussian", "agglomerative"])
        max_k = st.slider("Max clusters (KMeans / Gaussian)", 2, 15, 10)

        if st.button("📊 Run Clustering"):
            _maybe_balloons()
            with st.spinner("Clustering…"):
                try:
                    df_c = _ss.current_df.select_dtypes(include=[np.number]).copy()
                    df_clust, labels, _model, metrics = run_clustering(df_c, method=method, max_k=max_k)
                    st.success(f"Done. Silhouette score: {metrics['silhouette']:.3f}")
                    if "best_k" in metrics:
                        st.metric("Optimal K", metrics["best_k"])

                    pca = PCA(n_components=2)
                    comps = pca.fit_transform(df_clust.values)
                    plot_df = pd.DataFrame(comps, columns=["PC1", "PC2"])
                    plot_df["Cluster"] = labels.astype(str)
                    fig_cl = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster",
                                        title="Cluster Visualisation (PCA 2D)")
                    fig_cl.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
                    st.plotly_chart(fig_cl, use_container_width=True)
                    st.dataframe(df_clust.head(10))
                except Exception as exc:
                    st.error(f"Clustering failed: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: Forecasting
# ══════════════════════════════════════════════════════════════════════════════
if "📈 Forecasting" in tab_idx:
    with tabs[tab_idx["📈 Forecasting"]]:
        st.header("📈 Time Series Forecasting (Prophet)")

        all_cols = _ss.current_df.columns.tolist()
        date_col = st.selectbox("Date column", all_cols, key="fc_date")
        value_col = st.selectbox("Value column", all_cols, key="fc_val")
        horizon = st.slider("Forecast horizon (days)", 7, 365, 30)

        if st.button("📈 Run Forecast"):
            if date_col == value_col:
                st.error("Date and value columns must differ.")
            else:
                with st.spinner("Training Prophet model…"):
                    result = run_forecast(_ss.current_df, date_col, value_col, horizon)
                if result["success"]:
                    st.plotly_chart(result["plot"], use_container_width=True)
                    st.dataframe(result["forecast"][["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(15))
                    _ss.current_forecast = result
                else:
                    st.error(f"Forecast failed: {result['error']}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: RAG Chat
# ══════════════════════════════════════════════════════════════════════════════
with tabs[tab_idx["💬 RAG Chat"]]:
    st.header("💬 RAG Document Chat")
    st.markdown("Upload PDF, DOCX or TXT and ask questions. Uses local embeddings + Ollama.")

    ollama_ok = check_ollama()
    if not ollama_ok:
        st.warning("⚠️ Ollama not detected. Start with `ollama serve` and pull mistral.")

    doc_file = st.file_uploader("Upload document", type=["pdf", "docx", "txt"])
    if doc_file is not None:
        if st.button("📄 Index Document"):
            with st.spinner("Indexing…"):
                try:
                    _ss.rag_pipeline = RAGPipeline()
                    _ss.rag_pipeline.load_document(doc_file)
                    _ss.rag_pipeline.build_index()
                    st.success("Document indexed! Ask questions below.")
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")

    if _ss.rag_pipeline is not None:
        query = st.text_input("Ask a question about the document")
        if query:
            with st.spinner("Retrieving and generating…"):
                try:
                    answer, sources = _ss.rag_pipeline.query(query)
                    st.markdown(f"**Answer:** {answer}")
                    with st.expander("Source chunks"):
                        for i, src in enumerate(sources):
                            st.text(f"Chunk {i + 1}: {src[:300]}…")
                except Exception as exc:
                    st.error(f"RAG query failed: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: SQL Agent
# ══════════════════════════════════════════════════════════════════════════════
with tabs[tab_idx["🗄️ SQL Agent"]]:
    st.header("🗄️ Natural Language SQL Agent")
    st.markdown("Ask in plain English. The agent converts it to DuckDB SQL.")

    col_q, col_btn = st.columns([3, 1])
    with col_q:
        question = st.text_input("Ask in English", "Show the top 10 rows", key="sql_q")
    with col_btn:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        if st.button("🎤 Voice"):
            st.info("Voice input via Web Speech API (Chrome/Edge).")
            st.markdown("""
            <script>
            (function() {
                const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
                if (!SR) { alert('Speech recognition not supported in this browser.'); return; }
                const r = new SR();
                r.continuous = false; r.interimResults = false;
                r.onresult = function(e) {
                    const txt = e.results[0][0].transcript;
                    const inp = window.parent.document.querySelector('input[aria-label="Ask in English"]');
                    if (inp) { inp.value = txt; inp.dispatchEvent(new Event('input', {bubbles: true})); }
                };
                r.start();
            })();
            </script>""", unsafe_allow_html=True)

    sql_agent = SQLAgent()
    sql_agent.load_dataframe("data", _ss.current_df)

    if st.button("🔍 Execute Query"):
        with st.spinner("Generating SQL…"):
            result = sql_agent.ask(question)
        if result.get("success"):
            st.code(str(result.get("sql", "")), language="sql")
            st.dataframe(result.get("result"))
        else:
            st.warning(f"SQL Agent error: {result.get('error', 'unknown')}")
            if result.get("sql"):
                st.code(result.get("sql"), language="sql")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: Explainable AI
# ══════════════════════════════════════════════════════════════════════════════
with tabs[tab_idx["🔍 Explainable AI"]]:
    st.header("🔍 Explainable AI (SHAP)")
    st.markdown("Understand model decisions using SHAP values and permutation importance.")

    if st.button("🔍 Compute SHAP"):
        if _ss.current_model is None:
            st.warning("Train a model first (AutoML or Deep Learning tab).")
        else:
            with st.spinner("Computing SHAP values…"):
                shap_result = explain_model_shap(_ss.current_model, _ss.current_df, _ss.current_target)
            if shap_result.get("success"):
                st.pyplot(shap_result["summary_plot"])
                if shap_result.get("bar_plot") is not None:
                    st.pyplot(shap_result["bar_plot"])
            else:
                st.error(f"Explainability failed: {shap_result.get('error')}")

    # What-If sliders
    st.subheader("🎛 What-If Playground")
    st.caption("Adjust feature values and observe prediction changes.")
    if _ss.current_model is not None and _ss.current_df is not None:
        wif_df = _ss.current_df.copy()
        num_feats = wif_df.drop(columns=[_ss.current_target], errors="ignore") \
                          .select_dtypes(include=[np.number]).columns.tolist()[:5]
        sample_row = wif_df.head(1).copy()
        changed = False
        for feat in num_feats:
            mn = float(wif_df[feat].min())
            mx = float(wif_df[feat].max())
            cur = float(sample_row[feat].iloc[0])
            new_val = st.slider(f"{feat}", mn, mx, cur, key=f"wi_{feat}")
            if new_val != cur:
                sample_row[feat] = new_val
                changed = True
        if changed and hasattr(_ss.current_model, "predict"):
            try:
                pred = _ss.current_model.predict(sample_row)
                st.metric("Predicted value", f"{pred.values[0]}" if hasattr(pred, "values") else str(pred[0]))
            except Exception:
                pass
    else:
        st.info("Train a model to use the What-If playground.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: Multi-Agent
# ══════════════════════════════════════════════════════════════════════════════
with tabs[tab_idx["🤝 Multi-Agent"]]:
    st.header("🤝 Multi-Agent Pipeline")
    st.markdown("Explorer → Data Quality → Model → Critic → Business")

    if st.button("🚀 Run Multi-Agent"):
        with st.spinner("Running agents…"):
            results = run_multi_agent_pipeline(
                _ss.current_df, _ss.current_target, _ss.current_problem
            )
        for agent_name, output in results.items():
            with st.expander(f"📌 {agent_name}"):
                if isinstance(output, dict):
                    st.json(output)
                else:
                    st.write(output)

# ══════════════════════════════════════════════════════════════════════════════
# TAB: Goal Agent
# ══════════════════════════════════════════════════════════════════════════════
with tabs[tab_idx["🎯 Goal Agent"]]:
    st.header("🎯 Goal-Driven AutoML Agent")
    st.markdown("Set a target score. The agent searches hyperparameters to meet it.")

    target_score = st.number_input("Target score (0–1)", 0.50, 0.99, 0.85, 0.01)
    n_trials = st.slider("Optuna trials", 3, 20, 8)

    if st.button("🎯 Run Goal Agent"):
        with st.spinner("Optimising…"):
            res = run_goal_agent(
                _ss.current_df, _ss.current_target, _ss.current_problem,
                target_score=target_score, n_trials=n_trials,
            )
        st.metric("Best score", f"{res['best_score']:.4f}")
        st.metric("Goal met", "✅ Yes" if res["goal_met"] else "❌ No")
        st.json(res["best_params"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB: Business Analytics
# ══════════════════════════════════════════════════════════════════════════════
with tabs[tab_idx["📊 Business Analytics"]]:
    st.header("📊 Business Analytics Dashboard")

    if st.button("📊 Generate Insights"):
        with st.spinner("Analysing…"):
            insights = generate_business_insights(
                _ss.current_df, _ss.current_target, _ss.current_problem, _ss.current_model
            )
        st.subheader("📈 KPIs")
        st.json(insights["kpis"])
        st.subheader("📝 Executive Summary")
        st.markdown(insights["summary"])
        st.subheader("💡 Recommendations")
        for rec in insights["recommendations"]:
            st.markdown(f"- {rec}")
        st.subheader("⚠️ Risk Factors")
        for risk in insights["risks"]:
            st.markdown(f"- {risk}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: Reports
# ══════════════════════════════════════════════════════════════════════════════
with tabs[tab_idx["📄 Reports"]]:
    st.header("📄 Export Report")

    report_format = st.selectbox("Format", ["PDF", "DOCX"])
    include_charts = st.checkbox("Include charts", value=True)

    if st.button("📄 Generate Report"):
        with st.spinner("Building report…"):
            try:
                report_bytes = export_report(
                    _ss.current_df, _ss.current_target, _ss.current_problem,
                    model=_ss.current_model,
                    format=report_format.lower(),
                    include_charts=include_charts,
                )
                mime = "application/pdf" if report_format == "PDF" else \
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                st.download_button(
                    f"⬇️ Download {report_format}",
                    report_bytes,
                    f"apex_report.{report_format.lower()}",
                    mime=mime,
                )
            except Exception as exc:
                st.error(f"Report generation failed: {exc}")

    st.markdown("---")
    if st.button("📄 Auto-generate Executive Summary (PDF)"):
        with st.spinner("Generating executive summary PDF…"):
            try:
                exec_bytes = export_report(
                    _ss.current_df, _ss.current_target, _ss.current_problem,
                    model=_ss.current_model,
                    format="pdf",
                    include_charts=True,
                )
                st.download_button(
                    "⬇️ Download Executive Summary",
                    exec_bytes,
                    "executive_summary.pdf",
                    mime="application/pdf",
                )
            except Exception as exc:
                st.error(f"Executive summary failed: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: RL (optional)
# ══════════════════════════════════════════════════════════════════════════════
if "🎮 RL" in tab_idx:
    with tabs[tab_idx["🎮 RL"]]:
        st.header("🎮 Reinforcement Learning")

        _rl_available = False
        try:
            import gymnasium as gym  # type: ignore[import]
            from stable_baselines3 import PPO  # type: ignore[import]
            _rl_available = True
        except Exception:
            pass

        if not _rl_available:
            st.warning("gymnasium / stable-baselines3 not installed. Run: pip install gymnasium stable-baselines3")
        else:
            env_name = st.selectbox("Gymnasium Environment", ["CartPole-v1", "MountainCar-v0", "LunarLander-v2"])
            timesteps = st.slider("Training timesteps", 1000, 50000, 10000, 1000)

            if st.button("🎮 Train RL Agent"):
                _maybe_balloons()
                with st.spinner("Training RL agent…"):
                    try:
                        env = gym.make(env_name)
                        rl_model = PPO("MlpPolicy", env, verbose=0)
                        rl_model.learn(total_timesteps=timesteps)
                        obs, _ = env.reset()
                        done = False
                        total_reward = 0.0
                        steps = 0
                        while not done and steps < 1000:
                            action, _ = rl_model.predict(obs, deterministic=True)
                            obs, reward, terminated, truncated, _ = env.step(action)
                            done = terminated or truncated
                            total_reward += float(reward)
                            steps += 1
                        env.close()
                        st.success(f"✅ Training complete! Episode reward: {total_reward:.2f} in {steps} steps.")
                    except Exception as exc:
                        st.error(f"RL training failed: {exc}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer-text'>◆ Apex DS – Enterprise Analytics Studio · "
    "AutoGluon · PyTorch · Prophet · Local-first Architecture</div>",
    unsafe_allow_html=True,
)