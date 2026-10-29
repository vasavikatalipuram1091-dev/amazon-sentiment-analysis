import re
import string
from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(page_title="Amazon Review Sentiment Analysis", page_icon="💬", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def clean_review(text: str) -> str:
    """Match the notebook's clean_text steps: lowercase, remove punctuation/numbers, normalize spaces."""
    text = (text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


st.title("Amazon Review Sentiment Analysis")
st.write("Enter a customer review to predict whether the sentiment is Negative, Neutral, or Positive.")

if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    st.error("Model files are missing. Keep app.py, sentiment_model.pkl, and tfidf_vectorizer.pkl in the same folder.")
    st.stop()

try:
    model, tfidf = load_artifacts()
except Exception as exc:
    st.error("The model files could not be loaded. Check the scikit-learn version in requirements.txt and redeploy.")
    st.exception(exc)
    st.stop()

sample = st.selectbox(
    "Optional: try a sample review",
    [
        "Choose a sample (optional)",
        "Positive — The product is excellent and I am very happy with the quality.",
        "Negative — The product is very bad and I am disappointed.",
        "Neutral — It works fine but the quality is average.",
    ],
)

sample_text = sample.split(" — ", 1)[1] if " — " in sample else ""
review = st.text_area("Enter your Amazon review:", value=sample_text, height=140, placeholder="Type or paste a customer review here...")

if st.button("Predict Sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a review before predicting sentiment.")
    else:
        cleaned_review = clean_review(review)
        if not cleaned_review:
            st.warning("This review has no usable text after cleaning. Please enter a review containing words.")
        else:
            review_tfidf = tfidf.transform([cleaned_review])
            prediction = model.predict(review_tfidf)[0]
            st.subheader("Prediction")
            if prediction == "Positive":
                st.success(f"Predicted Sentiment: {prediction} 😊")
            elif prediction == "Negative":
                st.error(f"Predicted Sentiment: {prediction} 😞")
            else:
                st.info(f"Predicted Sentiment: {prediction} 😐")

            st.caption(f"Review statistics: {len(review)} characters · {len(review.split())} words")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(review_tfidf)[0]
                st.subheader("Class probabilities")
                probability_map = dict(zip(model.classes_, probabilities))
                for label in ["Negative", "Neutral", "Positive"]:
                    if label in probability_map:
                        score = float(probability_map[label])
                        st.write(f"**{label}: {score:.1%}**")
                        st.progress(score)
