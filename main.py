import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# API endpoint (bukan /docs ya 👇)
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
st.title("Sentiment Analysis Dashboard with API")

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
        st.write("Sedang melakukan labeling otomatis via API...")

        sentiments = []
        debug_prints = []  # buat simpan contoh response
        for idx, t in enumerate(df["text"]):
            try:
                response = requests.post(API_URL, json={"text": str(t)})
                if response.status_code == 200:
                    result = response.json()
                    # Simpan max 5 prediksi pertama untuk debug
                    if idx < 5:
                        debug_prints.append(result)
                    sentiments.append(result.get("prediction", "unknown"))
                else:
                    sentiments.append("unknown")
            except:
                sentiments.append("error")

        # Debug print: lihat 5 hasil API pertama
        st.subheader("Sample Predictions dari API")
        st.write(debug_prints)

        df["sentiment"] = sentiments
        st.write("Labeling selesai!")

        # === METRICS ===
        total = len(df)
        pos = (df["sentiment"] == "positif").sum()
        neg = (df["sentiment"] == "negatif").sum()
        neu = (df["sentiment"] == "netral").sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Amount of Data", total)
        col2.metric("Positive Count", pos)
        col3.metric("Negative Count", neg)
        col4.metric("Neutral Count", neu)

        # === PIE CHART ===
        st.subheader("Netral, Negatif, Positif")
        if total > 0 and (pos + neg + neu) > 0:
            fig1, ax1 = plt.subplots()
            ax1.pie(
                [pos, neg, neu],
                labels=["Positif", "Negatif", "Netral"],
                autopct='%1.1f%%',
                colors=["green", "red", "gray"]
            )
            st.pyplot(fig1)
        else:
            st.warning("Tidak ada data valid untuk ditampilkan di pie chart.")

        # === WORDCLOUD ===
        st.subheader("Wordcloud")
        text_data = " ".join(df["text"].astype(str).tolist())
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
        fig2, ax2 = plt.subplots()
        ax2.imshow(wordcloud, interpolation="bilinear")
        ax2.axis("off")
        st.pyplot(fig2)

        # === TOP WORDS PER SENTIMENT ===
        st.subheader("Most Common Words per Sentiment")
        for label in ["positif", "negatif", "netral"]:
            words = " ".join(df[df["sentiment"] == label]["text"].astype(str)).split()
            common_words = Counter(words).most_common(10)
            st.write(f"**{label.capitalize()}**: {common_words}")
