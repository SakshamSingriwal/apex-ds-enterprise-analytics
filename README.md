# Apex DS | Enterprise Analytics Studio

A local-first, offline enterprise analytics platform: AutoML, deep learning, RAG document chat, natural-language SQL, forecasting, clustering, explainable AI, multi-agent orchestration, and reporting — all in one Streamlit app. Runs entirely on your machine. No cloud. No paid APIs.

---

## ✨ Features

- **AutoML** (AutoGluon) — leaderboard of models, best one selected automatically.
- **Deep Learning** — ANN, CNN, LSTM, GRU, Transformer.
- **Clustering** — KMeans, DBSCAN, Gaussian Mixture, Agglomerative.
- **Time Series Forecasting** — Prophet.
- **RAG Document Chat** — chat over your own PDF/DOCX/TXT files, fully offline (Ollama).
- **Natural Language SQL Agent** — ask questions in English, get DuckDB queries and answers.
- **Explainable AI** — SHAP-based explanations for any trained model.
- **Multi-Agent System** — 5 coordinating agents for end-to-end analysis.
- **Goal Agent** — Optuna-driven hyperparameter/architecture search toward a stated goal.
- **Business Analytics Dashboard** — plain-English summaries and KPIs.
- **Report Export** — PDF/DOCX.
- **Autopilot** — a separate unattended CLI pipeline (see below) for hands-off batch runs.

---

## 📁 Structure

```
apex-ds-enterprise-analytics/
├── app.py                   # Streamlit entry point (interactive studio)
├── core/
│   ├── automl.py            # AutoGluon leaderboard + training
│   ├── business_insights.py # plain-English KPI summaries
│   ├── clustering.py        # KMeans / DBSCAN / GMM / Agglomerative
│   ├── deep_learning.py     # ANN / CNN / LSTM / GRU / Transformer
│   ├── feature_selection.py
│   ├── forecasting.py       # Prophet time-series forecasting
│   ├── goal_agent.py        # Optuna-driven goal-directed search
│   ├── multi_agent.py       # 5-agent orchestration
│   ├── multimodal.py
│   ├── preprocessing.py
│   ├── rag.py                # document chat (Ollama, local)
│   ├── reports.py            # PDF/DOCX export
│   ├── sql_agent.py          # natural-language -> DuckDB SQL
│   ├── utils.py
│   └── xai.py                # SHAP explainability
├── autopilot.py              # unattended CLI pipeline (see below)
├── autopilot_ui.py           # Streamlit UI for Autopilot runs
├── batch_runner.py           # runs Autopilot across multiple datasets
├── run_autopilot.bat
├── test_suite.py             # import/smoke checks across core modules
├── Telco_Customer_Churn.csv  # bundled demo dataset
├── requirements.txt
├── setup.bat / setup.sh
├── STARTUP_CHECKLIST.md      # offline / air-gapped install walkthrough
└── DEBUG_LOG.txt             # session-by-session bug-fix history
```

---

## 🚀 Quick start

### Windows
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama serve
streamlit run app.py
```
Or run `setup.bat` (requires Git Bash/WSL) to automate the above, including pulling the local Ollama model.

### Linux / WSL
```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
ollama serve &
streamlit run app.py
```

For a fully offline / air-gapped setup, see [STARTUP_CHECKLIST.md](STARTUP_CHECKLIST.md).

---

## 🗺️ Usage

1. Load the bundled demo dataset or upload your own CSV/XLSX file.
2. Train AutoML or custom deep learning models.
3. Use forecasting, clustering, explainability, the SQL agent, RAG document chat, and reporting.
4. Export trained models and reports locally.

---

## 🤖 Autopilot (unattended batch pipeline)

`autopilot.py` runs the entire workflow on any tabular dataset (CSV/XLSX/Parquet, up to ~100 GB) without the UI: profile/EDA → clean → encode → scale → imbalance handling → train/val/test split → model leaderboard → evaluation → SHAP → deployable model bundle → batch prediction. Large files are profiled with DuckDB (streamed, never fully loaded) and trained on a stratified sample.

```bash
python autopilot.py run data.csv                    # full pipeline, auto-detected target
python autopilot.py run data.csv --target Churn      # explicit target column
python autopilot.py run data.csv --time-budget 300   # cap training time (seconds)
python autopilot.py predict runs/<run_id> new.csv    # score new data with a saved run
python autopilot.py serve runs/<run_id>               # serve the model via FastAPI on :8000
```

`batch_runner.py` runs Autopilot across multiple datasets/roots in one go; `autopilot_ui.py` provides a Streamlit front end for browsing runs. Run output lands in `autopilot_runs/` (gitignored — it's generated data, not source).

---

## 🧪 Testing

```bash
python test_suite.py
```
Runs import and smoke checks across every `core/` module.

---

## 📄 License

See repository settings for license terms.
