
import streamlit as st
import os
import joblib
import re

# Page configuration
st.set_page_config(
    page_title="Clinical Trial Disease Classification",
    page_icon="🏥",
    layout="centered"
)

# Application title
st.title("🏥 Clinical Trial Disease Classification")

st.write(
    "Enter a clinical trial summary to predict its disease category."
)

# Model folder path
model_path = os.path.expanduser('~/Documents/clinic/models')

# Load trained model
loaded_model = joblib.load(
    os.path.join(model_path, 'disease_classifier.pkl')
)

# Load TF-IDF vectorizer
loaded_tfidf = joblib.load(
    os.path.join(model_path, 'tfidf_vectorizer.pkl')
)

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Disease prediction function
def predict_disease(text):

    cleaned_text = [preprocess_text(text)]

    text_tfidf = loaded_tfidf.transform(cleaned_text)

    prediction = loaded_model.predict(text_tfidf)

    return prediction[0]

# User input area
st.subheader("Enter Clinical Trial Summary")

user_input = st.text_area(
    "Clinical Trial Summary",
    placeholder="Enter the clinical trial summary here...",
    height=200
)

# Predict button
if st.button("Predict Disease"):

    if user_input.strip():

        result = predict_disease(user_input)

        st.success(f"Predicted Disease Category: {result}")

    else:
        st.warning("Please enter a clinical trial summary.")
