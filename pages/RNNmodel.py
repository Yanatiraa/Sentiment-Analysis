import os
import pickle
from tensorflow.keras.models import load_model
import streamlit as st

# Cache resources to improve performance
@st.cache_resource
def load_tokenizer_and_model():
    # Get the path to the parent directory of the `pages` folder
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

    # Define paths to the tokenizer and model files in the main directory
    tokenizer_path = os.path.join(base_dir, "tokenizer.pkl")
    model_path = os.path.join(base_dir, "rnn_sentiment_model.h5")

    # Check if files exist
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load tokenizer and model
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)
    model = load_model(model_path)

    return tokenizer, model

# Preprocessing function
def preprocess_text(text):
    import re
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text
