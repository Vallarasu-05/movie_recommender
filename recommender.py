import os
import zipfile
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

MODEL_DIR = "models"
ZIP_PATH = "models.zip"

# 🔽 DOWNLOAD + EXTRACT MODELS
def download_and_extract():
    if not os.path.exists(MODEL_DIR):
        st.info("⬇️ Downloading models...")

        # Make sure your zip is publicly accessible
        url = "https://drive.google.com/uc?id=1oeuasb6wO4uIF_ep7O3W0vYLx7tmyidu"
        try:
            import gdown
            gdown.download(url, ZIP_PATH, quiet=False)
        except Exception as e:
            st.error("Failed to download models. Check URL or permissions.")
            raise e

        st.info("📦 Extracting models...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(ZIP_PATH)
        st.success("✅ Models ready")

# 🔽 LOAD ARTIFACTS
@st.cache_resource(show_spinner=False)
def load_artifacts():
    download_and_extract()

    def _load(name):
        with open(os.path.join(MODEL_DIR, name), "rb") as f:
            return pickle.load(f)

    # Load TF-IDF matrices and indices
    tfidf_genres = _load("tfidf_genres.pkl")
    tfidf_keywords = _load("tfidf_keywords.pkl")
    tfidf_cast = _load("tfidf_cast.pkl")
    tfidf_overview = _load("tfidf_overview.pkl")
    indices = _load("indices.pkl")
    movies_master = pd.read_pickle(os.path.join(MODEL_DIR, "movies_master.pkl"))

    # Rebuild matrices
    matrix_genres = tfidf_genres.transform(movies_master['genres'].fillna(""))
    matrix_keywords = tfidf_keywords.transform(movies_master['keywords'].fillna(""))
    matrix_cast = tfidf_cast.transform(movies_master['cast'].fillna(""))
    matrix_overview = tfidf_overview.transform(movies_master['overview'].fillna(""))

    return {
        "indices": indices,
        "movies_master": movies_master,
        "matrix_genres": matrix_genres,
        "matrix_keywords": matrix_keywords,
        "matrix_cast": matrix_cast,
        "matrix_overview": matrix_overview
    }

# 🎯 CONTENT-BASED SIMILARITY
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

# 🔀 HYBRID (content-based only)
def recommend_hybrid(user_id, movie_title, top_n, arts):
    # Since CF is removed, hybrid = content
    return recommend_content(movie_title, top_n, arts)
