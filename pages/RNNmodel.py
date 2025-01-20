import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from spellchecker import SpellChecker
import os
from PIL import Image

# Load Model and Tokenizer
@st.cache_resource
def load_resources():
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    model = load_model("rnn_sentiment_model.h5")
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

    # Step 1: Product Selection
    st.header("Step 1: Choose the Product")
    product_types = ["Dresses", "Accessories", "Shoes", "Sportswear", "Cosmetics", "Jewellery", "Textiles", "Watches"]
    selected_product = st.selectbox("Select the product type:", product_types, index=0)

    # Step 2: Feature Selection
    st.header("Step 2: Feature Selection")
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

    # Step 3: Review Sentiment Analysis
    st.header("Step 3: Review Sentiment Analysis")
    user_review = st.text_input("Enter your review about the product (excluding negation and adjective words):")
    if user_review:
        corrected_review = correct_spelling(user_review)
        if corrected_review != user_review:
            st.warning(f"Did you mean: {corrected_review}?")

    # Step 4: Product Image Upload or Real-Time Capture
    st.header("Step 4: Upload Product Image or Take a Picture in Real-Time")
    mode = st.radio("Choose an option:", ("Upload Image", "Take a Picture"))
    
    uploaded_image = None
    captured_image = None

    if mode == "Upload Image":
        uploaded_image = st.file_uploader("Upload an image of the product:", type=["jpg", "jpeg", "png"])
    elif mode == "Take a Picture":
        captured_image = st.camera_input("Take a picture")

    # Submit Button
    if st.button("Submit"):
        # Validation
        if not selected_product:
            st.error("Please select a product type.")
        elif not any(user_choices.values()):
            st.error("Please select at least one feature.")
        elif not user_review:
            st.error("Please provide a review.")
        elif not (uploaded_image or captured_image):
            st.error("Please upload or capture an image of the product.")
        else:
            # Store data in product-specific database
            file_name = f"{selected_product}_choices.csv"
            if not os.path.exists(file_name):
                df = pd.DataFrame(columns=["Feature", "Choice", "Review"])
            else:
                df = pd.read_csv(file_name)

            for feature, choice in user_choices.items():
                if choice:
                    df = pd.concat([df, pd.DataFrame({"Feature": [feature], "Choice": [choice], "Review": [user_review]})])
            df.to_csv(file_name, index=False)
            st.success(f"Your selections for {selected_product} have been recorded!")

            # Display Recommendations and Reminders
            positive_features = df[df["Choice"].str.contains("Good|True|Nice|Suitable", na=False)]
            negative_features = df[df["Choice"].str.contains("Bad|Too Small|Too Large|Discomfort|Outdated|Unsuitable", na=False)]

            if not positive_features.empty:
                st.subheader("Recommendations:")
                st.write(positive_features["Choice"].value_counts().index[0])

            if not negative_features.empty:
                st.subheader("Reminders:")
                st.write(negative_features["Choice"].value_counts().index[0])

            # Perform Sentiment Analysis
            try:
                seq = tokenizer.texts_to_sequences([preprocess_text(user_review)])
                padded_seq = pad_sequences(seq, maxlen=100)
                prediction = model.predict(padded_seq)[0][0]
                sentiment = "Positive" if prediction > 0.5 else "Negative"
                st.subheader("Sentiment Analysis Result:")
                st.success(f"The sentiment of your review is: {sentiment}")
            except Exception as e:
                st.error(f"An error occurred during sentiment analysis: {e}")

            # Display Uploaded or Captured Image After Submission
            if uploaded_image:
                st.image(uploaded_image, use_container_width=True)
            elif captured_image:
                st.image(captured_image, use_container_width=True)

if __name__ == "__main__":
    main()
