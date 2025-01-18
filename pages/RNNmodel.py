import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from spellchecker import SpellChecker
import os

# Load Model and Tokenizer
@st.cache_resource
def load_resources():
    with open("tfidf_vectorizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    model = load_model("xgb_model.pkl")
    return tokenizer, model

# Preprocessing function
def preprocess_text(text):
    import re
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
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

    st.title("Sentiment Analysis of Fashion Product Reviews")

    # Section 1: Feature Selection
    st.header("Feature Selection")
    features = {
        "Sizing": ["True to Size", "Too Small", "Too Large"],
        "Quality": ["Good Quality", "Bad Quality"],
        "Comfort": ["Good Comfort", "Discomfort"],
        "Design": ["Nice Design", "Outdated Design"],
        "Functionality": ["Suitable", "Unsuitable"]
    }

    user_choices = {}
    for feature, options in features.items():
        include_feature = st.checkbox(f"Include {feature}", value=False)
        if include_feature:
            user_choices[feature] = st.radio(f"{feature}:", options, index=0)
        else:
            user_choices[feature] = None

    # Store user choices in a database (here using a CSV for simplicity)
    if st.button("Submit Feature Selection"):
        if not os.path.exists("user_choices.csv"):
            df = pd.DataFrame(columns=["Feature", "Choice"])
        else:
            df = pd.read_csv("user_choices.csv")

        for feature, choice in user_choices.items():
            if choice:
                df = pd.concat([df, pd.DataFrame({"Feature": [feature], "Choice": [choice]})])
        df.to_csv("user_choices.csv", index=False)
        st.success("Your feature choices have been recorded!")

    # Section 2: Recommendations and Reminders
    st.header("Recommendations and Reminders")
    if os.path.exists("user_choices.csv"):
        df = pd.read_csv("user_choices.csv")
        positive_features = df[df["Choice"].str.contains("Good|True|Nice|Suitable")]
        negative_features = df[df["Choice"].str.contains("Bad|Too Small|Too Large|Discomfort|Outdated|Unsuitable")]

        if not positive_features.empty:
            st.write("**Recommendations for Marketing:**")
            st.write(positive_features["Choice"].value_counts().index[0])

        if not negative_features.empty:
            st.write("**Reminders for Improvement:**")
            st.write(negative_features["Choice"].value_counts().index[0])

    # Section 3: Review Sentiment Analysis
    st.header("Review Sentiment Analysis")
    user_review = st.text_input("Enter your review about the product:")
    if user_review:
        corrected_review = correct_spelling(user_review)
        if corrected_review != user_review:
            st.warning(f"Did you mean: {corrected_review}?")

        if st.button("Analyze Sentiment"):
            try:
                seq = tokenizer.texts_to_sequences([preprocess_text(corrected_review)])
                padded_seq = pad_sequences(seq, maxlen=100)
                prediction = model.predict(padded_seq)[0][0]
                sentiment = "Positive" if prediction > 0.5 else "Negative"
                st.success(f"The sentiment of your review is: {sentiment}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Section 4: Product Image Upload
    st.header("Upload Product Image")
    uploaded_image = st.file_uploader("Upload an image of the product:", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Product Image", use_column_width=True)

if __name__ == "__main__":
    main()
