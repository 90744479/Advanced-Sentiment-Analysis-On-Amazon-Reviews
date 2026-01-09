import streamlit as st
import joblib
import re
import string

model = joblib.load('amazon_sentiment_model.pkl')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

st.title("📦 Amazon Sentiment Analyzer")
user_input = st.text_area("Enter review:")

if st.button("Predict"):
    if user_input:
        cleaned = clean_text(user_input)
        
        # Get probabilities for both classes
        # prob[0] is Negative, prob[1] is Positive
        prob = model.predict_proba([cleaned])[0]
        neg_score = prob[0]
        pos_score = prob[1]

        # LOGIC: If negative score is decent, show Negative
        # We use 0.4 instead of 0.5 to make it more sensitive to complaints
        if neg_score > 0.4:
            st.error(f"### Result: NEGATIVE")
            st.write(f"Negative Confidence: {neg_score*100:.1f}%")
        else:
            st.success(f"### Result: POSITIVE")
            st.write(f"Positive Confidence: {pos_score*100:.1f}%")