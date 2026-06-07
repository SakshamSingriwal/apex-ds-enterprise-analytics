# Apex DS | Enterprise Analytics Studio

Local-first enterprise analytics platform for AutoML, deep learning, RAG, SQL, forecasting, clustering, XAI, multi-agent orchestration, and reporting. Runs offline. No cloud. No paid APIs.

## Features

- AutoML (AutoGluon)
- Deep Learning (ANN, CNN, LSTM, GRU, Transformer)
- Clustering (KMeans, DBSCAN, Gaussian Mixture, Agglomerative)
- Time Series Forecasting (Prophet)
- RAG Document Chat (PDF/DOCX/TXT)
- Natural Language SQL Agent (DuckDB)
- Explainable AI (SHAP)
- Multi-Agent System (5 agents)
- Goal Agent (Optuna)
- Business Analytics Dashboard
- Report Export (PDF/DOCX)

## Installation (Windows)

1. Install Python 3.12 from python.org.
2. Open PowerShell in the project folder.
3. Run `setup.bat` (this requires Git Bash or WSL for `curl` and `bash`).
   - Alternatively, install Ollama manually from https://ollama.com, then run `ollama pull mistral`.
   - Then: `python -m venv .venv` and `pip install -r requirements.txt`.
4. Run: `.venv\Scripts\activate` then `ollama serve` then `streamlit run app.py`.

## Installation (Linux/WSL)

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
ollama serve &
streamlit run app.py
```

## Usage

1. Load a demo dataset or upload a CSV/XLSX file.
2. Train AutoML or custom deep learning models.
3. Use forecasting, clustering, explainability, SQL agent, RAG, and reporting features.
4. Export results and models locally.
