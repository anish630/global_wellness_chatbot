#!/bin/bash
# Setup script for Wellness Chatbot
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

echo "Setup complete! Run 'streamlit run app.py' to start the app."

