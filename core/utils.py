import logging
import subprocess
import sys
 
 
def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("apex_ds")
 
 
def check_ollama() -> bool:
    """Return True if Ollama is reachable."""
    try:
        import ollama  # type: ignore[import]
        ollama.list()
        return True
    except Exception:
        return False
 