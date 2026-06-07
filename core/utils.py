import logging
import subprocess
import sys
from pathlib import Path
import tempfile
import shutil
import hashlib
import streamlit as st
import pandas as pd
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)
def check_ollama():
    """Check if Ollama is running and mistral model is available."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        return 'mistral' in result.stdout
    except Exception:
        return False
@st.cache_data
def get_data_hash(df):
    """Generate hash of dataframe for caching."""
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
def cleanup_temp_files(directory: Path):
    """Remove temporary files older than 1 hour."""
    import time
    now = time.time()
    for f in directory.glob("*"):
        if f.is_file() and now - f.stat().st_mtime > 3600:
            f.unlink()
