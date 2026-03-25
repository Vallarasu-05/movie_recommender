# recommender.py

import os
import zipfile
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import gdown
import streamlit as st

# ── Paths ──
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
ZIP_PATH = os.path.join(BASE_DIR, "models.zip")

# ── DOWNLOAD + EXTRACT MODELS ──
def download_and_extract():
    if not os.path.exists(MODEL_DIR):
        st.info("⬇️ Downloading models...")
        url = "https://drive.google.com/uc?id=1oeuasb6wO4uIF_ep7O3W0vYLx7tmyidu"
        gdown.download(url, ZIP_PATH, quiet=False)

        st.info("📦 Extracting models...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)
        os.remove(ZIP_PATH)
        st.success("✅ Models ready")

# ── LOAD ARTIFACTS (CACHED) ──
@st.cache_resource(show_spinner=False)
def load_artifacts():
    download_and_extract()

    def _load(name):
        with open(os.path.join(MODEL_DIR, name), "rb") as f:
            return pickle.load(f)

    # Models
    svd = _load("svd_movie_model.pkl")
    tfidf_genres = _load("tfidf_genres.pkl")
    tfidf_keywords = _load("tfidf_keywords.pkl")
    tfidf_cast = _load("tfidf_cast.pkl")
    tfidf_overview = _load("tfidf_overview.pkl")
    indices = _load("indices.pkl")
    movies_master = pd.read_pickle(os.path.join(MODEL_DIR, "movies_master.pkl"))

    # Build TF-IDF matrices (cached for performance)
    matrix_genres = tfidf_genres.transform(movies_master['genres'].fillna(""))
    matrix_keywords = tfidf_keywords.transform(movies_master['keywords'].fillna(""))
    matrix_cast = tfidf_cast.transform(movies_master['cast'].fillna(""))
    matrix_overview = tfidf_overview.transform(movies_master['overview'].fillna(""))

    movie_ids = movies_master['movieId'].dropna().unique().astype(int)

    return {
        "svd": svd,
        "indices": indices,
        "movies_master": movies_master,
        "movie_ids": movie_ids,
        "matrix_genres": matrix_genres,
        "matrix_keywords": matrix_keywords,
        "matrix_cast": matrix_cast,
        "matrix_overview": matrix_overview
    }

# ── CONTENT-BASED SIMILARITY ──
def _content_similarity(idx, arts):
    return (
        linear_kernel(arts['matrix_genres'][idx], arts['matrix_genres']).flatten() * 3 +
        linear_kernel(arts['matrix_keywords'][idx], arts['matrix_keywords']).flatten() * 2 +
        linear_kernel(arts['matrix_cast'][idx], arts['matrix_cast']).flatten() * 1 +
        linear_kernel(arts['matrix_overview'][idx], arts['matrix_overview']).flatten() * 0.5
    )

def recommend_content(movie_title, top_n, arts):
    indices = arts['indices']
    movies = arts['movies_master']

    if movie_title not in indices.index:
        return []

    idx = indices[movie_title]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
    idx = int(idx)

    sim_scores = _content_similarity(idx, arts)
    ranked = np.argsort(sim_scores)[::-1]
    ranked = ranked[ranked != idx]

    return movies.iloc[ranked[:top_n]]['title'].tolist()

# ── COLLABORATIVE FILTERING ──
def recommend_cf(user_id, top_n, arts):
    svd = arts['svd']
    movie_ids = arts['movie_ids']
    movies = arts['movies_master']

    movie_lookup = (
        movies[['movieId', 'title']]
        .drop_duplicates('movieId')
        .set_index('movieId')
    )

    preds = [(mid, svd.predict(user_id, mid).est) for mid in movie_ids]
    preds.sort(key=lambda x: x[1], reverse=True)

    results = []
    for mid, _ in preds:
        if mid in movie_lookup.index:
            results.append(movie_lookup.loc[mid]['title'])
        if len(results) >= top_n:
            break

    return results

# ── HYBRID RECOMMENDER ──
def recommend_hybrid(user_id, movie_title, top_n, arts):
    cf = recommend_cf(user_id, top_n * 2, arts)
    content = recommend_content(movie_title, top_n * 2, arts)

    seen = set()
    results = []

    for m in cf + content:
        if m not in seen:
            seen.add(m)
            results.append(m)
        if len(results) >= top_n:
            break

    return results
