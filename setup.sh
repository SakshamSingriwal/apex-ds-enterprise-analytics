#!/usr/bin/env bash
set -e

echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh -o install_ollama.sh
bash install_ollama.sh
ollama pull mistral

echo "Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup complete!"
echo "To run: source .venv/bin/activate && ollama serve && streamlit run app.py"
