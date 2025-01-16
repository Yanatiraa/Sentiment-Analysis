import streamlit as st
import pickle
import tensorflow as tf
import numpy as np
from textblob import TextBlob
import pandas as pd
from PIL import Image

# Load pre-trained models
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)
model = tf.keras.models.load_model("rnn_sentiment_model.h5")

# Initialize a database (in memory for simplicity)
database = {
    "sizing": {"True to size": 0, "Too small": 0, "Too large": 0},
    "quality": {"Good quality": 0, "Bad quality": 0},
    "comfort": {"Good comfort": 0, "Discomfort": 0},
    "design": {"Nice design": 0, "Outdated design": 0},
    "functionality": {"Suitable": 0, "Unsuitable": 0},
}

# Functions for sentiment analysis
def predict_sentiment(review):
    sequences = tokenizer.texts_to_sequences([review])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
    prediction = model.predict(padded)
    if prediction > 0.6:
        return "Positive"
    elif prediction < 0.4:
        return "Negative"
    else:
        return "Neutral"

# Auto-correct functionality
def auto_correct(text):
    return str(TextBlob(text).correct())

# UI Design
st.title("Sentiment Analysis of Fashion Product Reviews")

# Feature selection
st.subheader("Select product features")
for category, options in database.items():
    st.write(category.capitalize())
    for option in options.keys():
        if st.checkbox(option):
            database[category][option] += 1

# Review input
st.subheader("Enter review about your product")
review = st.text_area("Write your review here")
if st.button("Submit Review"):
    corrected_review = auto_correct(review)
    sentiment = predict_sentiment(corrected_review)
    st.write(f"Corrected Review: {corrected_review}")
    st.write(f"Sentiment: {sentiment}")

# Image upload
st.subheader("Insert the product image")
uploaded_image = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Display results
st.subheader("Feature Summary")
feature_summary = pd.DataFrame(database).transpose()
st.table(feature_summary)

# Recommendations and reminders
st.subheader("Recommendations and Reminders")
positive_counts = {category: max(options, key=options.get) for category, options in database.items() if max(options.values()) > 0}
negative_counts = {category: min(options, key=options.get) for category, options in database.items() if min(options.values()) > 0}

if positive_counts:
    st.write("Recommendations for Marketing:")
    for category, feature in positive_counts.items():
        st.write(f"- {category.capitalize()}: Focus on {feature}.")

if negative_counts:
    st.write("Reminders for Improvement:")
    for category, feature in negative_counts.items():
        st.write(f"- {category.capitalize()}: Address issues with {feature}.")
