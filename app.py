# -*- coding: utf-8 -*-
"""
app.py - Streamlit prototype: Deteksi URL Phishing

Cara jalankan lokal:
    streamlit run app.py

Cara deploy publik (gratis):
    1. Push folder ini (app.py, detector_core.py, requirements.txt,
       dataset_phishing.csv, dan phishing_model.pkl jika sudah ada) ke repo GitHub.
    2. Buka https://share.streamlit.io -> "New app" -> pilih repo & file app.py.
    3. Selesai, dapat URL publik yang bisa diakses semua orang.
"""

import os
import time
import joblib

import pandas as pd
import streamlit as st

from detector_core import train, predict_url, MODEL_PATH, DATA_PATH

st.set_page_config(
    page_title="Deteksi URL Phishing",
    page_icon="🛡️",
    layout="centered"
)


# =========================================================
# LOAD / TRAIN MODEL (di-cache, hanya jalan sekali per server)
# =========================================================
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    if not os.path.exists(DATA_PATH):
        return None
    return train()


# =========================================================
# UI
# =========================================================
st.title("🛡️ Deteksi URL Phishing")
st.caption(
    "Prototype klasifikasi URL phishing vs legitimate menggunakan "
    "TF-IDF karakter + fitur leksikal URL, dilatih dengan RandomForest / "
    "Boosting / Stacking."
)

with st.spinner("Memuat model (training di first run bisa beberapa menit)..."):
    start = time.time()
    pipe = load_or_train_model()
    load_time = time.time() - start

if pipe is None:
    st.error(
        f"File `{DATA_PATH}` tidak ditemukan di repo, dan `{MODEL_PATH}` juga "
        "tidak ada. Sertakan salah satunya."
    )
    st.stop()

if load_time > 1:
    st.success(f"Model siap (dimuat/dilatih dalam {load_time:.1f} detik).")

st.divider()

url_input = st.text_input(
    "Masukkan URL yang ingin dicek:",
    placeholder="contoh: http://secure-paypal-login.verify-account.tk/update"
)

col1, col2 = st.columns([1, 4])
with col1:
    check = st.button("🔍 Cek URL", type="primary", use_container_width=True)

if check:
    if not url_input.strip():
        st.warning("Masukkan URL terlebih dahulu.")
    else:
        label, prob = predict_url(pipe, url_input.strip())

        if label == "PHISHING":
            st.error(f"⚠️ Terindikasi **PHISHING**")
        else:
            st.success(f"✅ Terindikasi **LEGITIMATE**")

        st.metric("Probabilitas phishing", f"{prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))

        with st.expander("Detail URL yang dicek"):
            st.code(url_input.strip())

st.divider()

with st.expander("ℹ️ Tentang prototype ini"):
    st.markdown(
        """
- Model dilatih dari dataset `dataset_phishing.csv` (11.430 URL, seimbang
  antara phishing & legitimate).
- Fitur yang dipakai: TF-IDF karakter n-gram dari URL + fitur leksikal
  (panjang URL, jumlah digit, entropy, ada IP, kata kunci seperti
  "login"/"secure"/"bank"/"paypal", dan lain-lain).
- Ini prototype untuk demonstrasi, **bukan** pengganti tools keamanan
  profesional. Selalu verifikasi URL mencurigakan lewat sumber resmi.
        """
    )

st.caption("Prototype — bukan untuk keputusan keamanan production tanpa validasi lebih lanjut.")
