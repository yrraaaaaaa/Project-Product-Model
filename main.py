import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud

API_URL = "http://127.0.0.1:8000/docs"

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

st.title("📊 Sentiment Analysis Dashboard with API")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Pastikan ada kolom teks
    if "text" not in df.columns:
        st.error("Dataset harus punya kolom 'text'")
    else:
        st.success("Dataset loaded!")

        # Panggil API untuk setiap teks 
        st.write("🔄 Sedang melakukan labeling otomatis via API...")

        sentiments = []
        for t in df["text"]:
            try:
                response = requests.post(API_URL, json={"text": str(t)})
                if response.status_code == 200:
                    sentiments.append(response.json()["prediction"])
                else:
                    sentiments.append("unknown")
            except:
                sentiments.append("error")

        df["sentiment"] = sentiments

        st.write("Labeling selesai!")

        # Metrics 
        total = len(df)
        pos = (df["sentiment"] == "positive").sum()
        neg = (df["sentiment"] == "negative").sum()
        neu = (df["sentiment"] == "neutral").sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Amount of Data", total)
        col2.metric("Positive Count", pos)
        col3.metric("Negative Count", neg)
        col4.metric("Neutral Count", neu)

        # Pie Chart 
        st.subheader(" Neutral, Negative, Positive")
        fig1, ax1 = plt.subplots()
        ax1.pie([pos, neg, neu], labels=["Positive", "Negative", "Neutral"], autopct='%1.1f%%', colors=["green", "red", "gray"])
        st.pyplot(fig1)

        # WordCloud 
        st.subheader("Wordcloud")
        text_data = " ".join(df["text"].astype(str).tolist())
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
        fig2, ax2 = plt.subplots()
        ax2.imshow(wordcloud, interpolation="bilinear")
        ax2.axis("off")
        st.pyplot(fig2)

        # Line Chart by Index 
        st.subheader("Sentiment Distribution Over Data Index")
        df["sentiment_score"] = df["sentiment"].map({"positive": 1, "neutral": 0, "negative": -1})
        st.line_chart(df["sentiment_score"])
