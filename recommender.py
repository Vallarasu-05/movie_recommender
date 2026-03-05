# recommender.py — Inference Engine
# Loads the exact pkl files already saved from your Colab notebook:
#   svd_movie_model.pkl, tfidf_genres.pkl, tfidf_keywords.pkl,
#   tfidf_cast.pkl, tfidf_overview.pkl, indices.pkl, movies_master.pkl
# ─────────────────────────────────────────────────────────────────────────────

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

MODEL_DIR = "model"


def load_artifacts():
    """Load all saved pkl files into a single dict for app.py to cache."""

    def _load(filename):
        with open(f"{MODEL_DIR}/{filename}", "rb") as f:
            return pickle.load(f)

    svd            = _load("svd_movie_model.pkl")
    tfidf_genres   = _load("tfidf_genres.pkl")
    tfidf_keywords = _load("tfidf_keywords.pkl")
    tfidf_cast     = _load("tfidf_cast.pkl")
    tfidf_overview = _load("tfidf_overview.pkl")
    indices        = _load("indices.pkl")
    movies_master  = pd.read_pickle(f"{MODEL_DIR}/movies_master.pkl")

    # Rebuild TF-IDF matrices from saved vectorizers + movies_master text columns
    matrix_genres   = tfidf_genres.transform(movies_master['genres'].fillna(""))
    matrix_keywords = tfidf_keywords.transform(movies_master['keywords'].fillna(""))
    matrix_cast     = tfidf_cast.transform(movies_master['cast'].fillna(""))
    matrix_overview = tfidf_overview.transform(movies_master['overview'].fillna(""))

    # Build movie_ids list from movies_master (no need for ratings at inference)
    movie_ids = movies_master['movieId'].dropna().unique().astype(int)

    return dict(
        svd=svd,
        indices=indices,
        movie_ids=movie_ids,
        movies_master=movies_master,
        matrix_genres=matrix_genres,
        matrix_keywords=matrix_keywords,
        matrix_cast=matrix_cast,
        matrix_overview=matrix_overview,
    )


# ── Content similarity ─────────────────────────────────────────────────────
def _content_similarity(idx, arts):
    return (
        linear_kernel(arts['matrix_genres'][idx],   arts['matrix_genres']).flatten()   * 3.0 +
        linear_kernel(arts['matrix_keywords'][idx], arts['matrix_keywords']).flatten() * 2.0 +
        linear_kernel(arts['matrix_cast'][idx],     arts['matrix_cast']).flatten()     * 1.0 +
        linear_kernel(arts['matrix_overview'][idx], arts['matrix_overview']).flatten() * 0.5
    )


def recommend_content(movie_title: str, top_n: int, arts: dict):
    indices       = arts['indices']
    movies_master = arts['movies_master']

    if movie_title not in indices.index:
        return []

    idx = indices[movie_title]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
    idx = int(idx)

    sim_scores = _content_similarity(idx, arts)
    ranked     = np.argsort(sim_scores)[::-1]
    ranked     = ranked[ranked != idx]
    ranked     = ranked[ranked < len(movies_master)]
    top        = ranked[:top_n]

    results = []
    for i in top:
        row = movies_master.iloc[i]
        results.append(_row_to_dict(row, score=round(float(sim_scores[i]), 3)))
    return results


# ── Collaborative filtering ────────────────────────────────────────────────
def recommend_cf(user_id: int, top_n: int, arts: dict):
    svd           = arts['svd']
    movie_ids     = arts['movie_ids']
    movies_master = arts['movies_master']

    movie_lookup = (
        movies_master[['movieId', 'title', 'genres', 'vote_average',
                        'release_date', 'poster_path', 'overview']]
        .drop_duplicates('movieId')
        .set_index('movieId')
    )

    preds = [(int(mid), svd.predict(user_id, int(mid)).est) for mid in movie_ids]
    preds.sort(key=lambda x: x[1], reverse=True)

    results = []
    for mid, score in preds[:top_n * 3]:
        if mid not in movie_lookup.index:
            continue
        row = movie_lookup.loc[mid]
        results.append(_row_to_dict(row, score=round(score, 2)))
        if len(results) >= top_n:
            break
    return results


# ── Hybrid ─────────────────────────────────────────────────────────────────
def recommend_hybrid(user_id: int, movie_title: str, top_n: int, arts: dict):
    cf      = recommend_cf(user_id, top_n * 2, arts)
    content = recommend_content(movie_title, top_n * 2, arts)

    seen, hybrid = set(), []
    for cf_item, cb_item in zip(cf, content):
        for item in [cf_item, cb_item]:
            if item['title'] not in seen:
                seen.add(item['title'])
                hybrid.append(item)
            if len(hybrid) >= top_n:
                return hybrid
    return hybrid[:top_n]


# ── Shared helper ──────────────────────────────────────────────────────────
def _row_to_dict(row, score):
    return {
        "title":        row['title'],
        "genres":       str(row.get('genres', '')),
        "overview":     str(row.get('overview', '')),
        "vote_average": row.get('vote_average', 'N/A'),
        "release_date": str(row.get('release_date', ''))[:4],
        "poster_path":  row.get('poster_path', ''),
        "score":        score,
    }
