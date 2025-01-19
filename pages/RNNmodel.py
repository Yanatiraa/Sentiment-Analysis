import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from spellchecker import SpellChecker
import os

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
    selected_product = st.selectbox("Select the product type:", product_types)

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
    user_review = st.text_input("Enter your review about the product:")
    corrected_review = None
    if user_review:
        corrected_review = correct_spelling(user_review)
        if corrected_review != user_review:
            st.warning(f"Did you mean: {corrected_review}?")

    # Step 4: Product Image Upload
    st.header("Step 4: Upload Product Image")
    uploaded_image = st.file_uploader("Upload an image of the product:", type=["jpg", "jpeg", "png"])

    # Submit Button
    if st.button("Submit"):
        if not selected_product:
            st.error("Please select a product type before you submit.")
        elif not any(user_choices.values()):
            st.error("Please select at least one feature before you submit.")
        elif not user_review:
            st.error("Please provide a review about the product before you submit.")
        elif not uploaded_image:
            st.error("Please upload the image of the product before you submit.")
        else:
            # Save user choices
            if not os.path.exists("user_choices.csv"):
                df = pd.DataFrame(columns=["Product", "Feature", "Choice", "Review"])
            else:
                df = pd.read_csv("user_choices.csv")

            for feature, choice in user_choices.items():
                if choice:
                    df = pd.concat([df, pd.DataFrame({"Product": [selected_product], "Feature": [feature], "Choice": [choice], "Review": [corrected_review]})])
            df.to_csv("user_choices.csv", index=False)

            st.success("Your submission has been recorded!")

            # Display Recommendations and Reminders
            st.header("Recommendations and Reminders")
            positive_features = df[df["Choice"].str.contains("Good|True|Nice|Suitable", na=False)]
            negative_features = df[df["Choice"].str.contains("Bad|Too Small|Too Large|Discomfort|Outdated|Unsuitable", na=False)]

            if not positive_features.empty:
                st.write(f"**Recommendations for {selected_product}:**")
                st.write(positive_features["Choice"].value_counts().index[0])

            if not negative_features.empty:
                st.write(f"**Reminders for {selected_product}:**")
                st.write(negative_features["Choice"].value_counts().index[0])

            # Display Sentiment Analysis Result
            seq = tokenizer.texts_to_sequences([preprocess_text(corrected_review)])
            padded_seq = pad_sequences(seq, maxlen=100)
            prediction = model.predict(padded_seq)[0][0]
            sentiment = "Positive" if prediction > 0.5 else "Negative"
            st.success(f"The sentiment of your review is: {sentiment}")

            # Display uploaded image
            st.image(uploaded_image, caption="Uploaded Product Image", use_column_width=True)

if __name__ == "__main__":
    main()
