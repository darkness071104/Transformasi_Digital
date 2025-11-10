import streamlit as st
import tensorflow as tf
import pickle
import re
import string
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Load model dan tokenizer ===
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("cnn_text_classifier.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# === Fungsi preprocessing ===
def clean_text(text):
    stop_words = set([
        'i','me','my','myself','we','our','ours','ourselves','you',
        'your','yours','yourself','yourselves','he','him','his','himself',
        'she','her','hers','herself','it','its','itself','they','them',
        'their','theirs','themselves','what','which','who','whom','this',
        'that','these','those','am','is','are','was','were','be','been',
        'being','have','has','had','having','do','does','did','doing','a',
        'an','the','and','but','if','or','because','as','until','while',
        'of','at','by','for','with','about','against','between','into',
        'through','during','before','after','above','below','to','from',
        'up','down','in','out','on','off','over','under','again','further',
        'then','once','here','there','when','where','why','how','all',
        'any','both','each','few','more','most','other','some','such',
        'no','nor','not','only','own','same','so','than','too','very',
        's','t','can','will','just','don','should','now'
    ])
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

# === Fungsi prediksi ===
def predict_ai_cnn(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
    prob = float(model.predict(pad)[0][0])
    return prob * 100

# === CONFIG UI ===
st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# === CSS THEME: CYBERPUNK ===
st.markdown("""
    <style>
    body {
        background-color: #0d0d0d;
        color: #e0e0e0;
    }
    .stApp {
        background: radial-gradient(circle at 20% 30%, #1a0033 0%, #000000 80%);
        border: 2px solid #6600ff;
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0px 0px 30px rgba(153, 0, 255, 0.6);
        max-width: 800px;
        margin: auto;
    }
    .title {
        text-align: center;
        color: #ff00ff;
        font-weight: 800;
        font-size: 2.5em;
        text-shadow: 0px 0px 10px #ff00ff, 0px 0px 25px #9900ff;
    }
    .subtitle {
        text-align: center;
        color: #a6a6a6;
        font-size: 1em;
        margin-bottom: 2em;
    }
    textarea {
        background-color: rgba(20, 20, 20, 0.8) !important;
        color: #e0e0e0 !important;
        border: 1px solid #ff00ff !important;
        border-radius: 10px !important;
    }
    .stButton button {
        background: linear-gradient(90deg, #ff00ff, #00ffff);
        color: black;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        box-shadow: 0 0 15px #00ffff;
        transition: 0.3s;
    }
    .stButton button:hover {
        box-shadow: 0 0 25px #ff00ff;
        transform: scale(1.05);
    }
    .result-card {
        border-radius: 15px;
        padding: 20px;
        color: white;
        font-weight: 600;
        text-align: center;
        margin-top: 15px;
        border: 1px solid #00ffff;
        box-shadow: 0 0 15px #00ffff;
        background: rgba(10, 10, 20, 0.8);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00ffff, #ff00ff);
    }
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #ff00ff, transparent);
    }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# === HEADER (statis) ===
st.markdown("<h1 class='title'>🤖 AI Text Detector 🤖</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Deteksi apakah teks ini hasil <b>AI</b> atau tulisan <b>manusia</b></p>", unsafe_allow_html=True)

# === INPUT (statis) ===
user_input = st.text_area("💬 Masukkan teks di bawah ini:", height=180, placeholder="Tulis teks di sini...")
analyze_button = st.button("🔍 Analisis Teks", use_container_width=True)

# === Placeholder untuk hasil (dinamis) ===
result_placeholder = st.empty()

# === BAGIAN HASIL DINAMIS ===
if analyze_button:
    if user_input.strip() == "":
        result_placeholder.warning("⚠️ Masukkan teks terlebih dahulu!")
    else:
        with result_placeholder.container():
            with st.spinner("🤖 Sedang menganalisis dengan jaringan CNN..."):
                prob = predict_ai_cnn(user_input)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("📊 Hasil Analisis:")

            st.progress(min(prob / 100, 1.0))
            st.write(f"**Kemungkinan teks ini dibuat oleh AI: {prob:.2f}%**")

            if prob > 80:
                st.markdown(f"<div class='result-card' style='border-color:#ff0066;box-shadow:0 0 20px #ff0066;'>🔴 <b>Sangat mungkin dibuat oleh AI</b></div>", unsafe_allow_html=True)
            elif prob > 50:
                st.markdown(f"<div class='result-card' style='border-color:#ffcc00;box-shadow:0 0 20px #ffcc00;'>🟠 <b>Mungkin dibuat oleh AI</b></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result-card' style='border-color:#00ff99;box-shadow:0 0 20px #00ff99;'>🟢 <b>Kemungkinan ditulis oleh manusia</b></div>", unsafe_allow_html=True)

# === FOOTER (statis) ===
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#888;'>Made with 👹 by <b>Kelompok 3</b> | Deteksi Teks AI</p>", unsafe_allow_html=True)
