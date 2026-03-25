# recommender.py
# ─────────────────────────────────────────────────────────────
# Core Recommendation Engine (Production Ready)
# Used by both FastAPI and Streamlit
# ─────────────────────────────────────────────────────────────

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

MODEL_DIR = "models"


# ─────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────
def load_artifacts():
    """Load all trained models and required data"""

    def load_pkl(filename):
        with open(f"{MODEL_DIR}/{filename}", "rb") as f:
            return pickle.load(f)

    # Load trained models
    svd            = load_pkl("svd_movie_model.pkl")
    tfidf_genres   = load_pkl("tfidf_genres.pkl")
    tfidf_keywords = load_pkl("tfidf_keywords.pkl")
    tfidf_cast     = load_pkl("tfidf_cast.pkl")
    tfidf_overview = load_pkl("tfidf_overview.pkl")

    # Load metadata
    indices        = load_pkl("indices.pkl")
    movies_master  = pd.read_pickle(f"{MODEL_DIR}/movies_master.pkl")

    # Ensure text fields are safe
    movies_master['genres']   = movies_master['genres'].fillna("")
    movies_master['keywords'] = movies_master['keywords'].fillna("")
    movies_master['cast']     = movies_master['cast'].fillna("")
    movies_master['overview'] = movies_master['overview'].fillna("")

    # 🔥 IMPORTANT: transform (not fit!)
    matrix_genres   = tfidf_genres.transform(movies_master['genres'])
    matrix_keywords = tfidf_keywords.transform(movies_master['keywords'])
    matrix_cast     = tfidf_cast.transform(movies_master['cast'])
    matrix_overview = tfidf_overview.transform(movies_master['overview'])

    # Movie IDs for CF
    movie_ids = movies_master['movieId'].dropna().unique().astype(int)

    return {
        "svd": svd,
        "indices": indices,
        "movies": movies_master,
        "movie_ids": movie_ids,
        "matrix_genres": matrix_genres,
        "matrix_keywords": matrix_keywords,
        "matrix_cast": matrix_cast,
        "matrix_overview": matrix_overview
    }


# ─────────────────────────────────────────────────────────────
# CONTENT-BASED SIMILARITY
# ─────────────────────────────────────────────────────────────
def _content_similarity(idx, arts):
    """Compute weighted similarity"""

    return (
        linear_kernel(arts['matrix_genres'][idx], arts['matrix_genres']).flatten() * 3.0 +
        linear_kernel(arts['matrix_keywords'][idx], arts['matrix_keywords']).flatten() * 2.0 +
        linear_kernel(arts['matrix_cast'][idx], arts['matrix_cast']).flatten() * 1.0 +
        linear_kernel(arts['matrix_overview'][idx], arts['matrix_overview']).flatten() * 0.5
    )


# ─────────────────────────────────────────────────────────────
# CONTENT-BASED RECOMMENDATION
# ─────────────────────────────────────────────────────────────
def recommend_content(movie_title: str, top_n: int, arts: dict):
    indices = arts['indices']
    movies  = arts['movies']

    if movie_title not in indices.index:
        return []

    idx = indices[movie_title]

    # 🔥 FIX: handle duplicate titles
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    idx = int(idx)

    sim_scores = _content_similarity(idx, arts)

    ranked = np.argsort(sim_scores)[::-1]
    ranked = ranked[ranked != idx]
    ranked = ranked[ranked < len(movies)]

    top_indices = ranked[:top_n]

    results = []
    for i in top_indices:
        row = movies.iloc[i]
        results.append(_row_to_dict(row, score=round(float(sim_scores[i]), 3)))

    return results


# ─────────────────────────────────────────────────────────────
# COLLABORATIVE FILTERING (SVD)
# ─────────────────────────────────────────────────────────────
def recommend_cf(user_id: int, top_n: int, arts: dict):
    svd    = arts['svd']
    movies = arts['movies']
    movie_ids = arts['movie_ids']

    movie_lookup = (
        movies[['movieId', 'title', 'genres', 'overview']]
        .drop_duplicates('movieId')
        .set_index('movieId')
    )

    predictions = []

    for mid in movie_ids:
        pred = svd.predict(user_id, int(mid)).est
        predictions.append((mid, pred))

    predictions.sort(key=lambda x: x[1], reverse=True)

    results = []
    for mid, score in predictions:
        if mid not in movie_lookup.index:
            continue

        row = movie_lookup.loc[mid]
        results.append(_row_to_dict(row, score=round(score, 2)))

        if len(results) >= top_n:
            break

    return results


# ─────────────────────────────────────────────────────────────
# HYBRID RECOMMENDATION
# ─────────────────────────────────────────────────────────────
def recommend_hybrid(user_id: int, movie_title: str, top_n: int, arts: dict):

    cf_results = recommend_cf(user_id, top_n * 2, arts)
    cb_results = recommend_content(movie_title, top_n * 2, arts)

    seen = set()
    final = []

    for item in cf_results + cb_results:
        if item['title'] not in seen:
            seen.add(item['title'])
            final.append(item)

        if len(final) >= top_n:
            break

    return final


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTION
# ─────────────────────────────────────────────────────────────
def _row_to_dict(row, score):
    return {
        "title": str(row.get('title', '')),
        "genres": str(row.get('genres', '')),
        "overview": str(row.get('overview', ''))[:150],
        "score": score
    }