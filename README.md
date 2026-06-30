# Deteksi URL Phishing — Streamlit App

## Isi folder
- `app.py` — aplikasi Streamlit (UI)
- `detector_core.py` — logic feature extraction, training, dan prediksi
- `dataset_phishing.csv` — dataset training (cadangan kalau model perlu dilatih ulang)
- `phishing_model.joblib` — model RandomForest yang sudah dilatih & dikompresi (~8.3 MB,
  akurasi ~92.5%, ROC-AUC ~0.977), supaya app langsung siap pakai tanpa training ulang
- `requirements.txt` — daftar dependency

## Jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```
Buka di browser: http://localhost:8501

## Deploy publik (gratis, via Streamlit Community Cloud)
1. Buat repo baru di GitHub, upload semua file di folder ini (termasuk `phishing_model.pkl`
   dan `dataset_phishing.csv`).
2. Buka https://share.streamlit.io, login dengan akun GitHub.
3. Klik **"New app"** → pilih repo tadi → branch `main` → file `app.py` → **Deploy**.
4. Tunggu beberapa menit, nanti dapat URL publik (format `https://namaapp.streamlit.app`)
   yang bisa diakses dan dipakai siapa saja.

## Alternatif deploy lain
- **Hugging Face Spaces** (gratis): pilih SDK "Streamlit" saat buat Space baru, upload file yang sama.
- **Railway / Render**: deploy sebagai web service Python biasa dengan start command
  `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

## Catatan
- Kalau `phishing_model.joblib` dihapus dari repo, app otomatis melatih ulang model dari
  `dataset_phishing.csv` saat pertama dibuka (sekitar 1-2 menit, hanya sekali per server
  karena di-cache dengan `st.cache_resource`).
- Model ini prototype/demo. Untuk produksi, pertimbangkan validasi lebih ketat, update
  dataset berkala, dan tambahan fitur seperti pengecekan reputasi domain real-time.
