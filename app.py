import streamlit as st
import joblib
import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model
model = joblib.load("xgboost_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Load your trained vectorizer

# Function to preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text.strip()

# Prediction function
def predict_sentiment(review):
    review_cleaned = clean_text(review)
    review_vectorized = vectorizer.transform([review_cleaned]).toarray()
    prediction = model.predict(review_vectorized)[0]

    # Reverse mapping to original sentiment labels
    reverse_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return reverse_mapping[prediction]

# Streamlit UI
st.title("Fashion Product Sentiment Analysis")

user_input = st.text_area("Enter a product review:", "")

if st.button("Analyze Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"**Predicted Sentiment:** {sentiment}")
    else:
        st.warning("Please enter a review to analyze.")
