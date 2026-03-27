import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import gzip
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# -------------------- Caching Model --------------------
@st.cache_resource
def load_model():
    with gzip.open("model.pkl.gz", "rb") as f:
        model = pkl.load(f)
    vect = pkl.load(open("vect.pkl", "rb"))
    return model, vect

model, vect = load_model()

# -------------------- Text Preprocessing --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)

    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]

    stemmer = SnowballStemmer("english")
    stems = [stemmer.stem(word) for word in words]

    return " ".join(stems)

# -------------------- UI --------------------
st.title("📰 Fake News Detection App")
st.markdown("""
Paste any news article or headline below and click **Check News** to verify whether it's **Fake** or **Legit**.
""")

news_text = st.text_area("✍️ Enter News Text Here", height=250, placeholder="Paste the news content here...")

# -------------------- Prediction --------------------
if st.button("🔍 Check News"):
    if news_text.strip() == "":
        st.warning("Please enter some news text first.")
    else:
        with st.spinner("Analyzing the news content..."):
            processed = clean_text(news_text)
            vect_text = vect.transform([processed])
            prediction = model.predict(vect_text)[0]

        st.markdown("---")
        st.subheader("🧾 Prediction Result")

        if prediction == 0:
            st.error("🚨 This News is **Fake News**")
        else:
            st.success("✅ This News is **Legit / Real**")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Built with using Streamlit | Machine Learning Fake News Classifier")
