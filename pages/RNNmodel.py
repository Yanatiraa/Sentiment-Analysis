import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from spellchecker import SpellChecker
import os
from PIL import Image
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Load Model and Tokenizer
@st.cache_resource
def load_resources():
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    model = load_model("rnn_sentiment_model.h5")
    return tokenizer, model

# Preprocessing function
def preprocess_user_input(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    stop = set(stopwords.words('english'))
    stemmer = SnowballStemmer("english")
    text = " ".join([stemmer.stem(word) for word in text.split() if word not in stop])
    return text

# Spell correction function
def correct_spelling(text):
    spell = SpellChecker()
    corrected = " ".join([spell.correction(word) for word in text.split()])
    return corrected

# Main Streamlit Application
def main():
    # Load tokenizer and model
    tokenizer, model = load_resources()

    st.title("Fashion Product Reviews Dashboard👗")

    # Step 3: Review Sentiment Analysis
    st.header("Step 3: Review Sentiment Analysis")
    user_review = st.text_input("Enter your review about the product:")
    if user_review:
        corrected_review = correct_spelling(user_review)
        if corrected_review != user_review:
            st.warning(f"Did you mean: {corrected_review}?")

    # Submit Button
    if st.button("Analyze Sentiment"):
        if not user_review:
            st.error("Please provide a review.")
        else:
            # Preprocess the user input
            cleaned_review = preprocess_user_input(user_review)

            # Tokenize and pad the user input
            user_sequence = tokenizer.texts_to_sequences([cleaned_review])
            user_padded = pad_sequences(user_sequence, maxlen=100)

            # Predict sentiment using the RNN model
            user_prediction = model.predict(user_padded).round()
            sentiment = "Positive" if user_prediction[0][0] == 1 else "Negative"

            st.subheader("Sentiment Analysis Result:")
            st.success(f"The sentiment of your review is: {sentiment}")

if __name__ == "__main__":
    main()
