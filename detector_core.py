# -*- coding: utf-8 -*-
"""
phishing_url_detector.py

Prototype deteksi URL phishing.
Berdasarkan pipeline dari notebook asli (TF-IDF char n-gram + fitur leksikal URL
+ RandomForest / XGBoost / Stacking), disederhanakan jadi satu script yang bisa:

  1. Melatih model dari dataset_phishing.csv
  2. Menyimpan model terbaik (.pkl)
  3. Masuk ke mode interaktif: ketik URL -> langsung dapat prediksi

Cara pakai:
    python phishing_url_detector.py

Catatan:
- Memakai pickle (bukan dill) supaya tidak butuh dependency tambahan.
  Karena itu FunctionTransformer dengan lambda diganti fungsi biasa
  (lambda tidak bisa di-pickle).
- Jika xgboost terpasang, otomatis dipakai. Kalau tidak ada,
  otomatis fallback ke GradientBoostingClassifier (built-in sklearn)
  supaya script tetap jalan di lingkungan mana pun.
"""

import re
import math
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from urllib.parse import urlparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False

DATA_PATH = "dataset_phishing.csv"
MODEL_PATH = "phishing_model.joblib"


# =========================================================
# 1. FEATURE EXTRACTOR (sama seperti notebook asli)
# =========================================================
class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.n_features_in_ = 1  # tandai sudah fitted (sklearn versi baru lebih strict)
        return self

    def _entropy(self, s):
        prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
        return -sum(p * math.log(p, 2) for p in prob) if len(s) > 0 else 0

    def _has_ip(self, s):
        return 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', s) else 0

    def _is_shortened(self, s):
        shortened_domains = ['bit.ly', 'goo.gl', 't.co', 'tinyurl.com', 'ow.ly', 'buff.ly', 'adf.ly']
        return 1 if any(sd in s for sd in shortened_domains) else 0

    def transform(self, X):
        X = np.asarray(X).astype(str)
        features = []
        for url in X:
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path

            len_url = len(url)
            len_domain = len(domain)
            len_path = len(path)
            num_digits = sum(c.isdigit() for c in url)
            num_upper = sum(c.isupper() for c in url)
            num_dots = url.count('.')
            num_hyphen = url.count('-')
            num_slash = url.count('/')
            num_question = url.count('?')
            num_equal = url.count('=')
            num_amp = url.count('&')
            num_percent = url.count('%')
            num_underscore = url.count('_')
            num_tilde = url.count('~')
            num_plus = url.count('+')

            contains_https_word = 1 if 'https' in url.lower() else 0
            contains_login = 1 if 'login' in url.lower() else 0
            contains_secure = 1 if 'secure' in url.lower() else 0
            contains_bank = 1 if 'bank' in url.lower() else 0
            contains_paypal = 1 if 'paypal' in url.lower() else 0

            tld_length = len(domain.split('.')[-1]) if '.' in domain else 0
            num_subdomain = len(domain.split('.')) - 2 if len(domain.split('.')) > 2 else 0
            path_depth = path.count('/')
            num_parameters = url.count('=')
            entropy_val = self._entropy(url)

            has_ip = self._has_ip(url)
            is_shortened = self._is_shortened(url)
            starts_with_http = 1 if url.lower().startswith('http') else 0
            ends_with_slash = 1 if url.endswith('/') else 0

            features.append([
                len_url, len_domain, len_path, num_digits, num_upper, num_dots, num_hyphen,
                num_slash, num_question, num_equal, num_amp, num_percent, num_underscore,
                num_tilde, num_plus, contains_https_word, contains_login, contains_secure,
                contains_bank, contains_paypal, tld_length, num_subdomain, path_depth,
                num_parameters, entropy_val, has_ip, is_shortened, starts_with_http, ends_with_slash
            ])

        return np.array(features)


# Nama tampilan (label) untuk tiap fitur, dipakai untuk menampilkan tabel di UI
FEATURE_DISPLAY_NAMES = [
    ("Panjang URL", "len_url", "int"),
    ("Panjang Domain", "len_domain", "int"),
    ("Panjang Path", "len_path", "int"),
    ("Jumlah Digit", "num_digits", "int"),
    ("Jumlah Huruf Kapital", "num_upper", "int"),
    ("Jumlah Titik", "num_dots", "int"),
    ("Jumlah Tanda Hubung (-)", "num_hyphen", "int"),
    ("Jumlah Garis Miring (/)", "num_slash", "int"),
    ("Jumlah Tanda Tanya (?)", "num_question", "int"),
    ("Jumlah Tanda Sama Dengan (=)", "num_equal", "int"),
    ("Jumlah Ampersand (&)", "num_amp", "int"),
    ("Jumlah Persen (%)", "num_percent", "int"),
    ("Jumlah Underscore (_)", "num_underscore", "int"),
    ("Jumlah Tilde (~)", "num_tilde", "int"),
    ("Jumlah Plus (+)", "num_plus", "int"),
    ("Mengandung kata 'https'", "contains_https_word", "bool"),
    ("Mengandung kata 'login'", "contains_login", "bool"),
    ("Mengandung kata 'secure'", "contains_secure", "bool"),
    ("Mengandung kata 'bank'", "contains_bank", "bool"),
    ("Mengandung kata 'paypal'", "contains_paypal", "bool"),
    ("Panjang TLD", "tld_length", "int"),
    ("Jumlah Subdomain", "num_subdomain", "int"),
    ("Kedalaman Path", "path_depth", "int"),
    ("Jumlah Parameter", "num_parameters", "int"),
    ("Entropy", "entropy", "float"),
    ("Mengandung Alamat IP", "has_ip", "bool"),
    ("URL Shortener", "is_shortened", "bool"),
    ("Dimulai dengan 'http'", "starts_with_http", "bool"),
    ("Diakhiri dengan '/'", "ends_with_slash", "bool"),
]


def get_url_features(url: str) -> dict:
    """Ekstrak fitur dari satu URL, dikembalikan sebagai dict {nama_tampilan: nilai}."""
    extractor = URLFeatureExtractor()
    extractor.fit([url])
    raw_values = extractor.transform([url])[0]  # array sesuai urutan num_feature_names_v2

    result = {}
    for (display_name, _key, dtype), value in zip(FEATURE_DISPLAY_NAMES, raw_values):
        if dtype == "bool":
            result[display_name] = "Ya" if value == 1 else "Tidak"
        elif dtype == "int":
            result[display_name] = int(value)
        else:  # float
            result[display_name] = round(float(value), 2)
    return result


def _get_url_column(d):
    """Pengganti lambda d: d['url'].values supaya bisa di-pickle."""
    return d['url'].values


def build_preprocessor():
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=1500)
    preprocessor = ColumnTransformer([
        ('tfidf', tfidf, 'url'),
        ('num', Pipeline([
            ('extract', FunctionTransformer(_get_url_column, validate=False)),
            ('urlfeat', URLFeatureExtractor())
        ]), ['url'])
    ], remainder='drop', sparse_threshold=0, n_jobs=1)
    return preprocessor


# =========================================================
# 2. TRAINING
# =========================================================
def train():
    print("Memuat dataset...")
    df = pd.read_csv(DATA_PATH)

    if not pd.api.types.is_numeric_dtype(df['status']):
        df['status'] = df['status'].astype(str).str.lower().map({'legitimate': 0, 'phishing': 1})

    X = df[['url']].copy()
    y = df['status'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {X_train.shape[0]} baris | Test: {X_test.shape[0]} baris")

    rf = RandomForestClassifier(n_estimators=150, class_weight='balanced', n_jobs=-1, random_state=42)

    if HAS_XGB:
        boost = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                               eval_metric='logloss', n_jobs=-1, random_state=42)
        boost_name = 'xgb'
    else:
        print("xgboost tidak terpasang, pakai GradientBoostingClassifier sebagai gantinya.")
        boost = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.15, random_state=42)
        boost_name = 'gb'

    stack = StackingClassifier(
        estimators=[('rf', rf), (boost_name, boost)],
        final_estimator=LogisticRegression(max_iter=1000),
        passthrough=True,
        n_jobs=-1
    )

    models = {
        'RandomForest': Pipeline([('preproc', build_preprocessor()), ('clf', rf)]),
        'Boosting': Pipeline([('preproc', build_preprocessor()), ('clf', boost)]),
        'Stacking': Pipeline([('preproc', build_preprocessor()), ('clf', stack)]),
    }

    results = []
    for name, pipe in models.items():
        print(f"\nTraining {name} ...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        res = {
            'name': name,
            'pipe': pipe,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
        }
        results.append(res)
        print(f"  Accuracy={res['accuracy']:.4f}  F1={res['f1']:.4f}  ROC-AUC={res['roc_auc']:.4f}")

    best = max(results, key=lambda r: r['f1'])
    print(f"\nModel terbaik: {best['name']} (F1={best['f1']:.4f})")

    joblib.dump(best['pipe'], MODEL_PATH, compress=9)
    print(f"Model disimpan ke: {MODEL_PATH}")

    return best['pipe']


# =========================================================
# 3. INTERAKTIF: CEK SATU URL
# =========================================================
def predict_url(pipe, url: str):
    X_input = pd.DataFrame({'url': [url]})
    pred = pipe.predict(X_input)[0]
    prob = pipe.predict_proba(X_input)[0][1]  # probabilitas kelas phishing
    label = "PHISHING" if pred == 1 else "LEGITIMATE"
    return label, prob


def interactive_loop(pipe):
    print("\n=== Mode interaktif: cek URL (ketik 'exit' untuk keluar) ===")
    while True:
        url = input("\nMasukkan URL: ").strip()
        if url.lower() in ('exit', 'quit', 'keluar'):
            print("Selesai.")
            break
        if not url:
            continue
        label, prob = predict_url(pipe, url)
        print(f"  -> Prediksi : {label}")
        print(f"  -> Probabilitas phishing : {prob:.2%}")
