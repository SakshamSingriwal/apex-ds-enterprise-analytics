import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import shutil
from sklearn.decomposition import PCA
import importlib.util

sys.path.insert(0, str(Path(__file__).parent))
from core.automl import get_feature_importance, train_automl
from core.business_insights import generate_business_insights
from core.deep_learning import get_available_architectures, train_deep_learning
from core.forecasting import run_forecast
from core.goal_agent import run_goal_agent
from core.multi_agent import run_multi_agent_pipeline
from core.rag import RAGPipeline
from core.reports import export_report
from core.sql_agent import SQLAgent
from core.utils import check_ollama, setup_logging
from core.xai import explain_model_shap
from core.clustering import run_clustering

st.set_page_config(
    page_title="Apex DS | Enterprise Analytics Studio",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = setup_logging()

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .stApp {
            background: linear-gradient(180deg, #0a0c10 0%, #12161d 100%);
        }
        .stSidebar {
            background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        }
        h1, h2, h3 {
            color: #58a6ff !important;
            font-weight: 600 !important;
        }
        .stButton>button {
            background: linear-gradient(135deg, #1f6feb, #388bfd);
            color: white;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(27, 127, 235, 0.2);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            background: linear-gradient(135deg, #388bfd, #50a1ff);
            box-shadow: 0 6px 16px rgba(27, 127, 235, 0.4);
        }
        .stMetric label {
            color: #8b949e !important;
            font-size: 14px !important;
        }
        .stMetric [data-testid="stMetricValue"] {
            color: #58a6ff !important;
            font-weight: 600 !important;
        }
        .stExpander {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 12px;
        }
        .stProgress > div > div {
            background-color: #39d353 !important;
        }
        .stAlert {
            border-radius: 8px;
        }
        .dataset-card {
            background: linear-gradient(135deg, #161b22 0%, #0d1117 0%);
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
            transition: all 0.3s ease;
        }
        .dataset-card:hover {
            border-color: #1f6feb;
        }
        .problem-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .classification-badge {
            background: linear-gradient(135deg, #39d353, #238636);
            color: #fff;
        }
        .regression-badge {
            background: linear-gradient(135deg, #f78166, #db4814);
            color: #fff;
        }
        .hero-text {
            text-align: center;
            padding: 60px 20px;
            color: #e6edf3;
        }
        .hero-title {
            font-size: 48px;
            font-weight: 700;
            background: linear-gradient(90deg, #58a6ff, #39d353);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero-subtitle {
            font-size: 20px;
            color: #8b949e;
            margin-top: 10px;
        }
        .logo-area {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 20px;
        }
        .logo-text {
            font-size: 24px;
            font-weight: 700;
            background: linear-gradient(90deg, #58a6ff, #39d353);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .sidebar-header {
            font-size: 20px;
            font-weight: 700;
            color: #58a6ff;
            margin-bottom: 16px;
            letter-spacing: 1px;
        }
        .footer-text {
            text-align: center;
            color: #8b949e;
            font-size: 12px;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #30363d;
        }
        .stTabs [data-baseweb="tab"] {
            transition: all 0.3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            transform: translateY(-1px);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'current_model_type' not in st.session_state:
    st.session_state.current_model_type = None
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'current_target' not in st.session_state:
    st.session_state.current_target = None
if 'current_problem' not in st.session_state:
    st.session_state.current_problem = None
if 'deep_learning_history' not in st.session_state:
    st.session_state.deep_learning_history = None
if 'multimodal_files' not in st.session_state:
    st.session_state.multimodal_files = []
if 'multimodal_preprocessed' not in st.session_state:
    st.session_state.multimodal_preprocessed = {}
if 'data_profile' not in st.session_state:
    st.session_state.data_profile = None

with st.sidebar:
    st.markdown("<div class='logo-area'><img src='https://img.icons8.com/fluency/96/artificial-intelligence.png' width='40'><span class='logo-text'>APEX ANALYTICS</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-header'>DATA SOURCE</div>", unsafe_allow_html=True)

    demo = st.radio(
        "Quick demo",
        ["None", "Iris (Classification)", "Churn (Classification)", "Housing (Regression)"],
    )
    uploaded = st.file_uploader("Or upload CSV/Excel", type=["csv", "xlsx"])

    if demo != "None":
        if demo == "Iris (Classification)":
            from sklearn.datasets import load_iris

            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["target"] = data.target
            target = "target"
        elif demo == "Churn (Classification)":
            np.random.seed(42)
            n = 1000
            df = pd.DataFrame(
                {
                    "tenure": np.random.randint(1, 72, n),
                    "monthly_charges": np.random.uniform(20, 120, n),
                    "age": np.random.randint(18, 70, n),
                    "contract": np.random.choice(["monthly", "yearly"], n),
                    "churn": np.random.binomial(1, 0.25, n),
                }
            )
            target = "churn"
        else:
            from sklearn.datasets import fetch_california_housing

            data = fetch_california_housing()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["target"] = data.target
            target = "target"

        st.session_state.current_df = df
        st.session_state.current_target = target
        problem = "classification" if pd.api.types.is_object_dtype(df[target]) or df[target].nunique() <= 20 else "regression"
        st.session_state.current_problem = problem
        st.success(f"Loaded {demo}")
    elif uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.current_df = df
            target = st.selectbox("Select target column", df.columns)
            st.session_state.current_target = target
            problem = "classification" if pd.api.types.is_object_dtype(df[target]) or df[target].nunique() <= 20 else "regression"
            st.session_state.current_problem = problem
            st.success("Data loaded")
        except Exception as error:
            st.error(f"Failed to load dataset: {error}")

    if st.session_state.current_df is not None:
        df = st.session_state.current_df
        st.markdown("<div class='dataset-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("📈 Features", len(df.columns))
        col3, col4 = st.columns(2)
        with col3:
            badge_class = "classification-badge" if st.session_state.current_problem == "classification" else "regression-badge"
            problem_text = (st.session_state.current_problem or "Unknown").capitalize()
            st.markdown(f"<span class='{badge_class}'>{'📊 ' + problem_text}</span>", unsafe_allow_html=True)
        with col4:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
            st.metric("🕳️ Missing %", f"{missing_pct:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("⚡ Enterprise Analytics Studio · Local-first · No cloud · Free forever")

st.title("Apex DS")
st.caption("Enterprise Analytics Studio · Professional AI platform for AutoML · Deep Learning · RAG · Multi-Agent")

if st.session_state.current_df is None:
    st.info("👈 Please load a dataset from the sidebar to begin.")
    st.stop()

tab_names = ["🤖 AutoML", "📊 Data Profiling"]

if importlib.util.find_spec("torch") is not None:
    tab_names.append("🧠 Deep Learning")

df = st.session_state.current_df
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
if len(numeric_cols) >= 2:
    tab_names.append("📊 Clustering")

has_datetime = False
datetime_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()
if not datetime_cols:
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col], errors='raise', infer_datetime_format=True)
                has_datetime = True
                break
            except Exception:
                pass
else:
    has_datetime = True
if has_datetime:
    tab_names.append("📈 Forecasting")

tab_names.extend(["💬 RAG Chat", "🗄️ SQL Agent", "🔍 Explainable AI"])
tab_names.extend(["🤝 Multi-Agent", "🎯 Goal Agent", "📊 Business Analytics", "📄 Reports"])

tabs = st.tabs(tab_names)

tab_idx = {name: i for i, name in enumerate(tab_names)}

# TAB 1: AutoML
with tabs[tab_idx["🤖 AutoML"]]:
    st.header("AutoML Engine")
    st.markdown("Automatically trains and compares multiple models. Supports binary classification, multiclass, and regression.")

    col1, col2 = st.columns([1, 2])
    with col1:
        time_limit = st.slider(
            "Training time limit (seconds)", 60, 600, 300, help="Longer training may improve accuracy"
        )
        if st.button("🚀 Run AutoML", type="primary"):
            with st.spinner("Training models... this may take a few minutes."):
                try:
                    predictor = train_automl(
                        st.session_state.current_df,
                        st.session_state.current_target,
                        st.session_state.current_problem,
                        time_limit=time_limit,
                    )
                    st.session_state.current_model = predictor
                    st.session_state.current_model_type = "automl"
                    st.success("Training complete!")
                except Exception as e:
                    st.error(f"AutoML failed: {e}")
                else:
                    if hasattr(predictor, '_fit_diagnostics'):
                        diag = predictor._fit_diagnostics
                        st.markdown("**Training Diagnostics**")
                        st.json({
                            'preset': diag.get('preset'),
                            'bag_folds': diag.get('bag_folds'),
                            'stack_levels': diag.get('stack_levels'),
                            'eval_metric': diag.get('eval_metric'),
                        })

                    try:
                        leaderboard = predictor.leaderboard(silent=True)
                    except Exception as le_err:
                        st.warning(f"Could not load leaderboard: {le_err}")
                        leaderboard = None

                    if leaderboard is not None and len(leaderboard) > 0:
                        st.subheader("🏆 Model Leaderboard")
                        st.dataframe(leaderboard[["model", "score_val", "fit_time", "pred_time_val"]].head(10))

                        best = leaderboard.iloc[0]
                        st.metric("Best Model", best["model"], f"Score: {best['score_val']:.4f}")

                        st.subheader("Feature Importance")
                        fi = get_feature_importance(predictor, st.session_state.current_df)
                        if fi is not None:
                            fig = px.bar(
                                fi.head(15),
                                x='importance',
                                y=fi.head(15).index,
                                orientation='h',
                                title='Top 15 Features',
                                color_discrete_sequence=['#58a6ff'] * len(fi.head(15))
                            )
                            fig.update_layout(template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#0d1117')
                            st.plotly_chart(fig, use_container_width=True)

                        with tempfile.TemporaryDirectory() as tmpdir:
                            predictor.save(tmpdir)
                            shutil.make_archive('model_export', 'zip', tmpdir)
                            with open('model_export.zip', 'rb') as f:
                                st.download_button('📦 Download Model (ZIP)', f, 'autogluon_model.zip')
    with col2:
        st.info(
            "💡 Tip: AutoGluon automatically handles missing values, encoding, and model stacking. It's one of the most accurate AutoML systems available."
        )

# TAB 2: Data Profiling
with tabs[tab_idx["📊 Data Profiling"]]:
    st.header("Data Profiling Suite")
    st.markdown("Comprehensive dataset analysis with statistical insights and visualizations.")

    df_profile = st.session_state.current_df
    target_col = st.session_state.current_target

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='dataset-card'><b>📊 Rows</b><br><span style='font-size:24px;'>{df_profile.shape[0]:,}</span></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='dataset-card'><b>📈 Columns</b><br><span style='font-size:24px;'>{len(df_profile.columns)}</span></div>", unsafe_allow_html=True)
    with col3:
        numeric_count = len(df_profile.select_dtypes(include=[np.number]).columns)
        st.markdown(f"<div class='dataset-card'><b>🔢 Numeric</b><br><span style='font-size:24px;'>{numeric_count}</span></div>", unsafe_allow_html=True)
    with col4:
        cat_count = len(df_profile.select_dtypes(include=['object', 'category']).columns)
        st.markdown(f"<div class='dataset-card'><b>🔤 Categorical</b><br><span style='font-size:24px;'>{cat_count}</span></div>", unsafe_allow_html=True)

    st.subheader("Correlation Heatmap")
    numeric_df = df_profile.select_dtypes(include=[np.number])
    if len(numeric_df.columns) >= 2:
        corr_matrix = numeric_df.corr()
        fig = px.imshow(
            corr_matrix,
            labels=dict(x="Features", y="Features", color="Correlation"),
            color_continuous_scale='RdBu',
            aspect="auto"
        )
        fig.update_layout(height=600, template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#0d1117')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough numeric columns for correlation analysis.")

    st.subheader("Missing Values Analysis")
    missing_data = df_profile.isnull().sum()
    missing_pct = (missing_data / len(df_profile) * 100).round(2)
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing %': missing_pct.values
    }).sort_values('Missing Count', ascending=False)

    if missing_data.sum() > 0:
        fig = px.bar(missing_df.head(20), x='Missing Count', y='Column', orientation='h', color='Missing Count', color_continuous_scale='reds')
        fig.update_layout(template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#0d1117', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(missing_df[missing_df['Missing Count'] > 0])
    else:
        st.success("✅ No missing values detected in the dataset!")

    st.subheader("Distribution Plots")
    dist_cols = df_profile.select_dtypes(include=[np.number]).columns.tolist()
    if dist_cols:
        selected_col = st.selectbox("Select feature for distribution", dist_cols)
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.histogram(df_profile, x=selected_col, nbins=50, title=f"Distribution of {selected_col}", marginal='box')
            fig.update_layout(template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#0d1117')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
            st.metric("Mean", f"{df_profile[selected_col].mean():.2f}")
            st.metric("Std", f"{df_profile[selected_col].std():.2f}")
            st.metric("Min", f"{df_profile[selected_col].min():.2f}")
            st.metric("Max", f"{df_profile[selected_col].max():.2f}")

# TAB 3: Deep Learning
if "🧠 Deep Learning" in tab_idx:
    with tabs[tab_idx["🧠 Deep Learning"]]:
        st.header("Deep Learning Lab")
        st.markdown("Train custom neural networks: ANN, 1D CNN, LSTM, GRU, Transformer")

        col1, col2 = st.columns([1, 1])
        with col1:
            arch = st.selectbox("Architecture", get_available_architectures())
            epochs = st.slider("Epochs", 10, 200, 80)
            batch_size = st.slider("Batch size", 16, 128, 32)
            learning_rate = st.select_slider(
                "Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3
            )
        with col2:
            dropout = st.slider("Dropout rate", 0.0, 0.7, 0.3, 0.05)
            patience = st.slider("Early stopping patience", 3, 30, 12)
            tune_hyperparameters = st.checkbox("Auto-tune hyperparameters (Optuna)", value=False)

        st.subheader("Feature Selection")
        st.caption("Built-in Mutual Information feature selection applied automatically")

        if st.button("🧠 Train Deep Learning Model", type="primary"):
            with st.spinner("Training neural network..."):
                model, history = train_deep_learning(
                    st.session_state.current_df,
                    st.session_state.current_target,
                    st.session_state.current_problem,
                    architecture=arch,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    patience=patience,
                    tune_hyperparameters=tune_hyperparameters,
                    dropout=dropout,
                )
            st.session_state.current_model = model
            st.session_state.deep_learning_history = history

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(y=history['train_loss'], name='Train Loss', line=dict(color='#58a6ff'))
            )
            fig.add_trace(
                go.Scatter(y=history['val_loss'], name='Validation Loss', line=dict(color='#3fb950'))
            )
            fig.update_layout(
                title='Training Curves',
                xaxis_title='Epoch',
                yaxis_title='Loss',
                template='plotly_dark',
                paper_bgcolor='#0d1117',
                plot_bgcolor='#0d1117'
            )
            st.plotly_chart(fig, use_container_width=True)

            final_gap = abs(history['train_loss'][-1] - history['val_loss'][-1])
            if final_gap > 0.2:
                st.warning(
                    f"⚠️ Possible overfitting detected (validation gap = {final_gap:.3f}). Consider increasing dropout or reducing epochs."
                )
            else:
                st.success(f"✅ Model well-fitted (validation gap = {final_gap:.3f})")

            st.subheader("Model Diagnostics")
            st.markdown(f"**Architecture:** {history.get('architecture', arch)}")
            st.markdown(f"**Input dimensions:** {history.get('input_dim', 'N/A')}")
            st.markdown(f"**Output dimensions:** {history.get('output_dim', 'N/A')}")
            st.markdown(f"**Problem type:** {history.get('problem_type', st.session_state.current_problem)}")

# TAB 4: Clustering
if "📊 Clustering" in tab_idx:
    with tabs[tab_idx["📊 Clustering"]]:
        st.header("Clustering Studio")
        st.markdown("Unsupervised clustering with automatic K selection and PCA visualization.")

        method = st.selectbox(
            "Clustering algorithm",
            ['kmeans', 'dbscan', 'gaussian_mixture', 'agglomerative'],
        )
        if st.button("🔍 Run Clustering"):
            with st.spinner("Clustering..."):
                df_clust, labels, model, metrics = run_clustering(
                    st.session_state.current_df, method=method
                )
            st.success(f"Clustering complete. Silhouette score: {metrics['silhouette']:.3f}")
            if 'best_k' in metrics:
                st.metric("Optimal number of clusters", metrics['best_k'])

            X = df_clust.select_dtypes('number').values
            pca = PCA(2)
            pca_result = pca.fit_transform(X)
            plot_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
            plot_df['cluster'] = labels.astype(str)
            fig = px.scatter(
                plot_df,
                x='PC1',
                y='PC2',
                color='cluster',
                title='Cluster Visualization (PCA)',
            )
            fig.update_layout(template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#0d1117')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_clust.head())

# TAB 5: Forecasting
if "📈 Forecasting" in tab_idx:
    with tabs[tab_idx["📈 Forecasting"]]:
        st.header("Time Series Forecasting (Prophet)")
        st.markdown("Forecast future values using Facebook Prophet (handles seasonality and trends).")

        if st.session_state.current_df is None:
            st.warning("Please load a dataset before using forecasting.")
            st.stop()

        date_cols = st.session_state.current_df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()
        if not date_cols:
            date_cols = [col for col in st.session_state.current_df.columns]
        value_cols = st.session_state.current_df.select_dtypes(include=[np.number]).columns.tolist()
        if not value_cols:
            value_cols = [col for col in st.session_state.current_df.columns]

        date_col = st.selectbox("Date column", date_cols, key='forecast_date_col')
        value_col = st.selectbox("Value column", value_cols, key='forecast_value_col')
        horizon = st.slider("Forecast horizon (days)", 7, 365, 30)

        if st.button("📈 Run Forecast"):
            if date_col == value_col:
                st.error("Date column and value column must be different.")
            else:
                with st.spinner("Training Prophet model..."):
                    result = run_forecast(
                        st.session_state.current_df,
                        date_col,
                        value_col,
                        horizon,
                    )
                if result['success']:
                    st.plotly_chart(result['plot'], use_container_width=True)
                    st.dataframe(
                        result['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
                    )
                    st.session_state.current_forecast = result
                else:
                    st.error(f"Forecast failed: {result['error']}")

# TAB 6: RAG Chat
with tabs[tab_idx["💬 RAG Chat"]]:
    st.header("RAG Document Chat")
    st.markdown("Upload PDF, DOCX, or TXT files and ask questions. Uses local embeddings and Ollama.")

    ollama_ok = check_ollama()
    if not ollama_ok:
        st.warning(
            "⚠️ Ollama is not running or mistral model not pulled. Please start Ollama and run: `ollama pull mistral`"
        )

    uploaded_doc = st.file_uploader("Upload document", type=['pdf', 'docx', 'txt'])
    if uploaded_doc and ollama_ok:
        if st.button("📄 Index Document"):
            with st.spinner("Indexing..."):
                st.session_state.rag_pipeline = RAGPipeline()
                st.session_state.rag_pipeline.load_document(uploaded_doc)
                st.session_state.rag_pipeline.build_index()
            st.success("Document indexed! You can now ask questions.")

    if st.session_state.rag_pipeline is not None and ollama_ok:
        query = st.text_input("Ask a question about the document")
        if query:
            with st.spinner("Retrieving and generating..."):
                answer, sources = st.session_state.rag_pipeline.query(query)
            st.markdown(f"**Answer:** {answer}")
            with st.expander("Source chunks"):
                for i, src in enumerate(sources):
                    st.text(f"Chunk {i+1}: {src[:200]}...")

# TAB 7: SQL Agent
with tabs[tab_idx["🗄️ SQL Agent"]]:
    st.header("Natural Language SQL Agent")
    st.markdown("Ask questions in plain English, and the agent converts them to SQL using DuckDB.")

    sql_agent = SQLAgent()
    sql_agent.load_dataframe('data', st.session_state.current_df)

    question = st.text_input("Ask in English", "Show top 10 rows")
    if st.button("🔍 Execute"):
        with st.spinner("Generating SQL..."):
            result = sql_agent.ask(question)
        if result['success']:
            st.code(result['sql'], language='sql')
            st.dataframe(result['result'])
        else:
            st.error(f"Error: {result['error']}")

# TAB 8: Explainable AI
with tabs[tab_idx["🔍 Explainable AI"]]:
    st.header("Explainable AI (SHAP)")
    st.markdown("Understand model predictions using SHAP values.")

    if st.button("🔍 Compute SHAP Explanations"):
        if st.session_state.current_model is None:
            st.warning("Please train a model first in the AutoML or Deep Learning tab.")
        elif not hasattr(st.session_state.current_model, 'predict'):
            st.warning("SHAP explanations are only supported for AutoGluon models in this demo.")
        else:
            with st.spinner("Computing SHAP values (this may take a while)..."):
                shap_result = explain_model_shap(
                    st.session_state.current_model,
                    st.session_state.current_df,
                    st.session_state.current_target,
                )
            if shap_result['success']:
                st.pyplot(shap_result['summary_plot'])
                st.pyplot(shap_result['bar_plot'])
            else:
                st.warning(f"SHAP not available: {shap_result['error']}")

# TAB 9: Multi-Agent System
with tabs[tab_idx["🤝 Multi-Agent"]]:
    st.header("Multi-Agent Pipeline")
    st.markdown("Orchestrated AI agents: Explorer → Data Quality → Model → Critic → Business")

    if st.button("🚀 Run Multi-Agent Pipeline"):
        with st.spinner("Running agents..."):
            results = run_multi_agent_pipeline(
                st.session_state.current_df,
                st.session_state.current_target,
                st.session_state.current_problem,
            )
        for agent_name, output in results.items():
            with st.expander(f"📌 {agent_name}"):
                if isinstance(output, dict):
                    st.json(output)
                else:
                    st.write(output)

# TAB 10: Goal Agent
with tabs[tab_idx["🎯 Goal Agent"]]:
    st.header("Goal-Driven AutoML Agent")
    st.markdown("Specify a target score, and the agent automatically tunes hyperparameters to reach it.")
    st.caption("Uses robust cross-validated settings to minimize overfitting.")

    target_score = st.number_input("Target score (0-1)", 0.70, 0.99, 0.85, 0.01)
    n_trials = st.slider("Number of Optuna trials", 3, 20, 8)

    if st.button("🎯 Run Goal Agent"):
        with st.spinner("Optimizing..."):
            result = run_goal_agent(
                st.session_state.current_df,
                st.session_state.current_target,
                st.session_state.current_problem,
                target_score=target_score,
                n_trials=n_trials,
            )
        st.metric("Best score achieved", f"{result['best_score']:.4f}")
        st.metric("Goal met", "✅ Yes" if result['goal_met'] else "❌ No")
        st.json(result['best_params'])

# TAB 11: Business Analytics
with tabs[tab_idx["📊 Business Analytics"]]:
    st.header("Business Analytics Dashboard")
    st.markdown("Automatically generate KPIs, executive summary, and actionable recommendations.")

    if st.button("📊 Generate Business Insights"):
        with st.spinner("Analyzing..."):
            insights = generate_business_insights(
                st.session_state.current_df,
                st.session_state.current_target,
                st.session_state.current_problem,
                st.session_state.current_model,
            )
        st.subheader("📈 KPIs")
        st.json(insights['kpis'])
        st.subheader("📝 Executive Summary")
        st.markdown(insights['summary'])
        st.subheader("💡 Recommendations")
        for rec in insights['recommendations']:
            st.markdown(f"- {rec}")
        st.subheader("⚠️ Risk Factors")
        for risk in insights['risks']:
            st.markdown(f"- {risk}")

# TAB 12: Reports
with tabs[tab_idx["📄 Reports"]]:
    st.header("Export Report")
    st.markdown("Generate a comprehensive report in PDF or DOCX format.")

    report_format = st.selectbox("Format", ["PDF", "DOCX"])
    include_charts = st.checkbox("Include charts", value=True)

    if st.button("📄 Generate Report"):
        with st.spinner("Generating report..."):
            report_bytes = export_report(
                st.session_state.current_df,
                st.session_state.current_target,
                st.session_state.current_problem,
                model=st.session_state.current_model,
                format=report_format.lower(),
                include_charts=include_charts,
            )
        st.download_button(
            "Download Report",
            report_bytes,
            f"report.{report_format.lower()}",
            mime="application/pdf" if report_format == "PDF" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

st.markdown("<div class='footer-text'>Apex DS | Enterprise Analytics Studio · Powered by AutoGluon, PyTorch, Prophet · Local-first Architecture</div>", unsafe_allow_html=True)