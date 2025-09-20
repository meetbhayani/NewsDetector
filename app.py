import subprocess
import sys

# Install heavy packages at runtime if not already installed
def install_runtime_packages():
    packages = [
        "torch==2.2.2",
        "transformers==4.56.2"
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", package])

try:
    import torch
    import transformers
except ImportError:
    install_runtime_packages()

# Now safe to import transformers
from transformers import BertTokenizer, BertForSequenceClassification


import streamlit as st
from sentence_transformers import SentenceTransformer , util
import pandas as pd
import torch

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# Load model/tokenizer once
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("./saved_fake_news_model")
    tokenizer = BertTokenizer.from_pretrained("./saved_fake_news_model")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# Load real news reference
@st.cache_resource
def load_reference_data():
    df = pd.read_csv('E:/WORK/fakenews/real-news.csv')
    print("Columns:", df.columns)  # Debugging line
    embedder = SentenceTransformer('all-MiniLM-L6-v2') 
    embeddings = embedder.encode(df['real_news'].tolist(), convert_to_tensor=True)  # Adjusted
    return df, embedder, embeddings

real_news_df, embedder, real_embeddings = load_reference_data()

# Streamlit UI

st.title("📰 Fake News Detector")
st.markdown("Enter a news statement or headline to check if it's **real** or **fake**.")

text = st.text_area("📝 News Statement:", height=150)

if st.button("🔍 Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
            prediction = torch.argmax(logits, dim=1).item()

        label_map = {0: "❌ Fake News", 1: "✅ Real News"}
        st.success(f"**Prediction:** {label_map[prediction]}")

        # Show related real news if prediction is fake
        if prediction == 0:
            st.info("🔎 Searching for related verified news...")
            input_embedding = embedder.encode(text, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(input_embedding, real_embeddings)
            top_idx = torch.argmax(cosine_scores).item()
            related_news = real_news_df.iloc[top_idx]['real_news']
            st.success(f"🗞 **Related Verified News:**\n\n{related_news}")
