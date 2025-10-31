import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go

# --- Download stopwords jika belum ada
nltk.download('stopwords')

# --- Load data
@st.cache_data
def load_data():
    df = pd.read_csv("balanced_ai_human_prompts.csv")
    return df

df = load_data()

# --- Preprocessing function
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df["clean_text"] = df["text"].apply(clean_text)
df["number_of_words"] = df["text"].apply(lambda x: len(x.split()))
df["number_of_char"] = df["text"].apply(lambda x: len(x))

# --- Caching model training biar gak retrain terus
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["generated"],
        test_size=0.2, random_state=42, stratify=df["generated"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=300))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "conf_matrix": confusion_matrix(y_test, y_pred)
    }

    return model, metrics

model, metrics = train_model()

# --- Fungsi prediksi
def predict_ai_percentage(text):
    cleaned = clean_text(text)
    prob = model.predict_proba([cleaned])[0][1]
    return round(prob * 100, 2)

# --- Streamlit UI
st.title("🧠 Deteksi Teks AI vs Human")
st.write("Model ini memprediksi kemungkinan apakah teks ditulis oleh manusia atau AI berdasarkan analisis linguistik dan pola bahasa.")
st.markdown("---")
# --- Sidebar
with st.sidebar:
    st.title("🔍 Menu Navigasi")
    menu = st.selectbox(
        "Pilih Tampilan:",
        [
            "Data Overview",
            "Distribusi Data",
            "Evaluasi Model",
            "Prediksi Teks"
        ]
    )

# --- Menu 1: Data Overview
if menu == "Data Overview":
    st.header("📋 Data Overview")
    st.write(df.head(10))
    st.markdown("---")
    st.write("Jumlah data:", len(df))
    st.markdown("---")
    st.markdown("### Contoh Data Bersih")
    st.write(df[["text", "clean_text"]].head(5))

# --- Menu 2: Distribusi Data
elif menu == "Distribusi Data":
    st.header("📊 Distribusi Label dan Statistik Teks")

    fig, ax = plt.subplots()
    sns.countplot(data=df, x="generated", palette="Set2", ax=ax)
    ax.set_title("Distribusi Label (AI vs Human)")
    st.pyplot(fig)
    st.markdown("---")
    st.header("📊 Statistik Panjang Teks")
    st.dataframe(df[["number_of_words", "number_of_char"]].describe())

    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df["number_of_words"], bins=30, ax=ax2[0], color="skyblue")
    ax2[0].set_title("Distribusi Jumlah Kata")
    sns.histplot(df["number_of_char"], bins=30, ax=ax2[1], color="lightgreen")
    ax2[1].set_title("Distribusi Jumlah Karakter")
    st.pyplot(fig2)

# --- Menu 3: Evaluasi Model
elif menu == "Evaluasi Model":
    st.header("📈 Evaluasi Kinerja Model")
    st.markdown("---")

    # Tampilan metrik utama dalam 3 kolom
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Akurasi", f"{metrics['accuracy']*100:.2f}%", "Performa keseluruhan")
    with col2:
        st.metric("⚖️ F1 Score", f"{metrics['f1']:.2f}", "Keseimbangan Precision & Recall")
    with col3:
        st.metric("💡 ROC-AUC", f"{metrics['roc_auc']:.2f}", "Kemampuan klasifikasi")

    st.markdown("---")
    st.subheader("📊 Classification Report")

    # Styling dataframe agar lebih enak dilihat
    df_report = pd.DataFrame(metrics["report"]).transpose()
    st.dataframe(
        df_report.style.background_gradient(cmap="Blues").format(precision=2)
    )

# --- Menu 4: Prediksi Teks
elif menu == "Prediksi Teks":
    st.header("🤖 Predeksi Teks AI")
    user_input = st.text_area("Masukkan teks yang ingin diuji:", height=200)

    if st.button("Prediksi"):
        if user_input.strip() == "":
            st.warning("Masukkan teks terlebih dahulu!")
        else:
            prob = predict_ai_percentage(user_input)

            # Buat diagram lingkaran
            fig = go.Figure(go.Pie(
                values=[prob, 100 - prob],
                labels=["AI Generated", "Human Written"],
                hole=0.7,
                textinfo='none',
                marker_colors=["#C93DFF", "#EAEAEA"]
            ))

            # Tampilkan persentase di tengah
            fig.update_layout(
                annotations=[dict(
                    text=f"<b>{prob:.0f}%</b>",
                    x=0.5, y=0.5, font_size=40, showarrow=False
                )],
                showlegend=False,
                height=350,
                margin=dict(t=0, b=0, l=0, r=0)
            )

            # Tampilkan grafik di Streamlit
            st.plotly_chart(fig, use_container_width=True)

            # Keterangan tambahan
            if prob > 80:
                st.error(f"🔴 Teks ini kemungkinan **{prob:.0f}%** dibuat oleh AI.")
            else:
                st.success(f"🟢 Teks ini kemungkinan **{prob:.0f}%** ditulis oleh manusia.")