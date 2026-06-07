@echo off
echo Installing Ollama...
curl -fsSL https://ollama.com/install.sh -o install_ollama.sh
bash install_ollama.sh
ollama pull mistral
echo Creating Python virtual environment...
python -m venv .venv
call .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
echo Setup complete!
echo To run: .venv\Scripts\activate && ollama serve && streamlit run app.py

