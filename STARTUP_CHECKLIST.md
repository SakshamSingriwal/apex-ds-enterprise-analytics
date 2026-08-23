# Apex DS — Offline / Air-Gap Startup Checklist

## Windows

### 1. Install Python 3.12
- Download from https://www.python.org/downloads/release/python-31210/
- Check "Add Python to PATH" during install.
- Verify: `python --version` → `Python 3.12.x`

### 2. Create Virtual Environment
```cmd
cd C:\Users\SAKSHAM SINGRIWAL\Desktop\apex-ds-enterprise-analytics
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```cmd
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Offline / Air-Gap Install (Optional)
- Pre-download wheels to a `wheelhouse` folder on an internet-connected machine:
  ```powershell
  pip download -r requirements.txt -d wheelhouse --platform win_amd64 --python-version 312 --only-binary=:all:
  ```
- Transfer `wheelhouse` to the air-gapped machine.
- Install from local cache:
  ```cmd
  pip install -r requirements.txt --no-index --find-links wheelhouse
  ```
- If some packages lack pure-Python wheels, use a trusted PyPI mirror URL:
  ```cmd
  pip install -r requirements.txt --index-url https://mirror.example.com/simple
  ```

### 5. Install & Start Ollama (RAG Backend)
- Download Ollama from https://ollama.com/download/windows
- Start Ollama (runs as a background service).
- Pull the required model:
  ```cmd
  ollama pull mistral
  ```

### 6. Start the App
```cmd
streamlit run app.py --server.headless true
```

> **GPU is not required.** All heavy libraries (`torch==2.2.0`, `autogluon>=1.5.0`, `faiss-cpu==1.7.4`) have CPU-compatible wheelhouse entries. CPU-only training will be slower but functional.

---

## WSL / Linux

### 1. Install Python 3.12
```bash
sudo apt update && sudo apt install python3.12 python3.12-venv python3-pip
```

### 2. Create & Activate Virtual Environment
```bash
cd /mnt/c/Users/SAKSHAM SINGRIWAL/Desktop/apex-ds-enterprise-analytics
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Air-Gap Caveats
- `faiss-cpu` and `torch` have manylinux Wheels; if the target distro is Alpine or musl-based, you may need to transpile or use a conda-forge cache.
- `opencv-python` may require system `libGL` (`sudo apt install libgl1-mesa-glx`).
- If running behind a proxy, set:
  ```bash
  export HTTP_PROXY=http://proxy:8080
  export HTTPS_PROXY=http://proxy:8080
  ```

### 5. Start Ollama
```bash
# On Linux/WSL
ollama serve &
ollama pull mistral
```

### 6. Start the App
```bash
streamlit run app.py --server.headless true
```

---

## Validation

Run these checks inside the activated venv to confirm everything works:

```bash
python test_runtime.py
python run_demo.py
streamlit run app.py --server.headless true
```

---

## Troubleshooting Checklist

### Ollama Not Running
- Check if service is up: `ollama list`
- Restart: `ollama serve` or `ollama run mistral`
- Verify `check_ollama()` in `core/utils.py` returns `True`.
- If port 11434 is already in use, kill the conflicting process or set `OLLAMA_HOST=0.0.0.0:11435`.

### AutoGluon `fit()` Kwargs Compatibility
- AutoGluon 1.1.1 removed or renamed several kwargs (e.g., `auto_stack`, `num_bag_folds`).
- If training crashes, pin to presets only: use `TabularPredictor.fit(..., presets="medium_quality")` or downgrade to `autogluon==0.8.2`.
- Check `core/automl.py` and update `train_automl` to match the installed version's signature.

### Tab Visibility
- **Deep Learning (🧠)** tab appears only if `torch` is importable.
- **Clustering (📊)** tab appears if dataset has fewer than 2 numeric columns.
- **Forecasting (📈)** tab appears only if a datetime column is detected.
- If a tab is missing, ensure `requirements.txt` installed cleanly and the dataset meets the column-type thresholds defined in `app.py` lines 282–304.

### Common Pip Issues
- Slow or failing installs on air-gapped machines? Point to a local wheelhouse or internal mirror:
  ```cmd
  pip install -r requirements.txt --no-index --find-links wheelhouse
  ```
- Exclude broken packages if needed and install piecewise:
  ```cmd
  pip install streamlit pandas numpy scikit-learn autogluon torch sentence-transformers faiss-cpu ollama
  ```
