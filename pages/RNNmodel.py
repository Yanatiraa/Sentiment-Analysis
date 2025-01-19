import streamlit as st
import pandas as pd
import pickle
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib  # Use joblib for XGBoost model
import os

# Load Model and TfidfVectorizer
@st.cache_resource
def load_resources():
    with open("tokenizer_xgb.pkl", "rb") as handle:  # Load TfidfVectorizer
        vectorizer = pickle.load(handle)
    model = joblib.load("xgboost_sentiment_model.pkl")  # Load XGBoost model
    return vectorizer, model

# Preprocessing function
def preprocess_text(text):
    import re
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    return text

# Spell correction function
def correct_spelling(text):
    spell = SpellChecker()
    corrected = " ".join([spell.correction(word) for word in text.split()])
    return corrected

# Main Streamlit Application
def main():
    # Load vectorizer and model
    vectorizer, model = load_resources()

    st.title("Sentiment Analysis of Fashion Product Reviews")

    # Section 1: Review Sentiment Analysis
    st.header("Review Sentiment Analysis")
    user_review = st.text_input("Enter your review about the product:")
    if user_review:
        corrected_review = correct_spelling(user_review)
        if corrected_review != user_review:
            st.warning(f"Did you mean: {corrected_review}?")

        if st.button("Analyze Sentiment"):
            try:
                # Preprocess and transform the input text
                processed_review = preprocess_text(corrected_review)
                vectorized_review = vectorizer.transform([processed_review])  # Correctly transform using TfidfVectorizer
                prediction = model.predict(vectorized_review)[0]  # Get the prediction
                sentiment = "Positive" if prediction == 1 else "Negative"  # Map the prediction to sentiment
                st.success(f"The sentiment of your review is: {sentiment}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Section 2: Product Image Upload
    st.header("Upload Product Image")
    uploaded_image = st.file_uploader("Upload an image of the product:", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Product Image", use_column_width=True)

if __name__ == "__main__":
    main()
