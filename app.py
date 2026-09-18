import streamlit as st
import numpy as np
import re
import pickle
import tensorflow as tf

from gensim.models import Word2Vec
from tensorflow import keras


# =========================================================
# 1. LOAD SAVED MODELS
# =========================================================

word2vec_model = Word2Vec.load("models/word2vec.model")

model = keras.models.load_model(
    "models/sentiment_model.keras",
    compile=False
)

with open("models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("models/label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)


# =========================================================
# 2. TEXT CLEANING FUNCTION
# =========================================================

def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags
    text = re.sub(r"#\w+", "", text)

    # Keep only alphabets and spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# 3. CONVERT TEXT INTO WORD2VEC VECTOR
# =========================================================

def get_sentence_vector(tokens):

    vectors = [
        word2vec_model.wv[word]
        for word in tokens
        if word in word2vec_model.wv
    ]

    # If no words are found in Word2Vec vocabulary
    if len(vectors) == 0:
        return np.zeros(200)

    # Take the average of all word vectors
    return np.mean(vectors, axis=0)


# =========================================================
# 4. PREDICTION FUNCTION
# =========================================================

def predict_sentiment(text):

    # Clean text
    cleaned_text = clean_text(text)

    # Tokenize
    tokens = cleaned_text.split()

    # Convert tweet into 200-dimensional vector
    sentence_vector = get_sentence_vector(tokens)

    # Convert to 2D array
    sentence_vector = np.array(sentence_vector).reshape(1, -1)

    # Scale using the saved scaler
    scaled_vector = scaler.transform(sentence_vector)

    # Predict using neural network
    prediction_probability = model.predict(
        scaled_vector,
        verbose=0
    )

    # Get predicted encoded class
    predicted_class = np.argmax(prediction_probability, axis=1)[0]

    # Convert encoded class back to original label
    predicted_label = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    return predicted_label, prediction_probability[0]


# =========================================================
# 5. STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Apple Twitter Sentiment Analysis",
    page_icon="🍎",
    layout="centered"
)


# =========================================================
# 6. APPLICATION TITLE
# =========================================================

st.title("🍎 Apple Twitter Sentiment Analysis")

st.write(
    "Analyze the sentiment of an Apple-related tweet "
    "using Word2Vec and Deep Learning."
)


# =========================================================
# 7. TEXT INPUT
# =========================================================

tweet = st.text_area(
    "Enter an Apple-related tweet:",
    placeholder="Example: I love my new iPhone!"
)


# =========================================================
# 8. PREDICT BUTTON
# =========================================================

if st.button("Predict Sentiment"):

    if tweet.strip() == "":
        st.warning("Please enter a tweet.")

    else:

        predicted_label, probabilities = predict_sentiment(tweet)

        # Convert numerical labels into readable names
        sentiment_names = {
            -1: "Negative",
            0: "Neutral",
            1: "Positive"
        }

        sentiment = sentiment_names.get(
            predicted_label,
            str(predicted_label)
        )

        # Display result
        st.subheader("Prediction")

        if sentiment == "Positive":
            st.success("😊 Positive Sentiment")

        elif sentiment == "Negative":
            st.error("😞 Negative Sentiment")

        else:
            st.info("😐 Neutral Sentiment")

        # Display original tweet
        st.write("**Tweet:**")
        st.write(tweet)

        # Display cleaned tweet
        cleaned = clean_text(tweet)

        st.write("**Cleaned Text:**")
        st.write(cleaned)

        # Display prediction probabilities
        st.subheader("Prediction Probabilities")

        probability_data = {
            "Negative": float(probabilities[0]),
            "Neutral": float(probabilities[1]),
            "Positive": float(probabilities[2])
        }

        st.bar_chart(probability_data)