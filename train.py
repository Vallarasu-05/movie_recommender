# train.py
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: Since you already have all trained pkl files from Google Colab,
# you only need to copy them into the model/ folder. This script is only
# needed if you want to RETRAIN from scratch using the raw CSV files.
#
# YOUR EXISTING FILES (copy these to model/):
#   svd_movie_model.pkl   → model/svd_movie_model.pkl
#   tfidf_genres.pkl      → model/tfidf_genres.pkl
#   tfidf_keywords.pkl    → model/tfidf_keywords.pkl
#   tfidf_cast.pkl        → model/tfidf_cast.pkl
#   tfidf_overview.pkl    → model/tfidf_overview.pkl
#   indices.pkl           → model/indices.pkl
#   movies_master.pkl     → model/movies_master.pkl
#
# ─────────────────────────────────────────────────────────────────────────────

import os, ast, pickle, logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR  = "data"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load raw CSVs ──────────────────────────────────────────────────────────
log.info("Loading CSVs...")
credits     = pd.read_csv(f"{DATA_DIR}/credits.csv")
keywords    = pd.read_csv(f"{DATA_DIR}/keywords.csv")
links_big   = pd.read_csv(f"{DATA_DIR}/links.csv")
movies      = pd.read_csv(f"{DATA_DIR}/movies_metadata.csv", low_memory=False)
ratings_big = pd.read_csv(f"{DATA_DIR}/ratings_1M.csv")

# ── Clean & merge ──────────────────────────────────────────────────────────
movies['id']        = pd.to_numeric(movies['id'], errors='coerce')
movies              = movies.dropna(subset=['id'])
movies['id']        = movies['id'].astype(int)
links_big['tmdbId'] = pd.to_numeric(links_big['tmdbId'], errors='coerce')
links_big           = links_big.dropna(subset=['tmdbId'])
links_big['tmdbId'] = links_big['tmdbId'].astype(int)

movies_master = movies.merge(links_big, left_on='id', right_on='tmdbId')
movies_master = movies_master.merge(keywords, on='id')
movies_master = movies_master.merge(credits, on='id')
movies_master = movies_master.reset_index(drop=True)
log.info("movies_master: %s", movies_master.shape)

# ── Filter ratings ─────────────────────────────────────────────────────────
valid_movies = movies_master['movieId']
ratings_cf   = ratings_big[ratings_big['movieId'].isin(valid_movies)]
ratings_cf   = ratings_cf[ratings_cf['rating'] >= 2].copy()
log.info("Filtered ratings: %s", ratings_cf.shape)

# ── SVD ────────────────────────────────────────────────────────────────────
log.info("Training SVD...")
reader   = Reader(rating_scale=(0.5, 5))
data     = Dataset.load_from_df(ratings_cf[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
svd = SVD(n_factors=150, lr_all=0.005, reg_all=0.05, n_epochs=30, random_state=42)
svd.fit(trainset)
preds = svd.test(testset)
log.info("RMSE: %.4f | MAE: %.4f",
         accuracy.rmse(preds, verbose=False),
         accuracy.mae(preds,  verbose=False))

# ── TF-IDF ─────────────────────────────────────────────────────────────────
def extract_names(text):
    try:    return " ".join([i['name'] for i in ast.literal_eval(text)])
    except: return ""

def extract_cast(text):
    try:    return " ".join([i['name'] for i in ast.literal_eval(text)[:5]])
    except: return ""

movies_master['genres']   = movies_master['genres'].apply(extract_names).fillna("")
movies_master['keywords'] = movies_master['keywords'].apply(extract_names).fillna("")
movies_master['cast']     = movies_master['cast'].apply(extract_cast).fillna("")
movies_master['overview'] = movies_master['overview'].fillna("")

tfidf_genres   = TfidfVectorizer(stop_words='english', max_features=500)
tfidf_keywords = TfidfVectorizer(stop_words='english', max_features=500)
tfidf_cast     = TfidfVectorizer(stop_words='english', max_features=500)
tfidf_overview = TfidfVectorizer(stop_words='english', max_features=1000)

tfidf_genres.fit_transform(movies_master['genres'])
tfidf_keywords.fit_transform(movies_master['keywords'])
tfidf_cast.fit_transform(movies_master['cast'])
tfidf_overview.fit_transform(movies_master['overview'])

indices = pd.Series(movies_master.index, index=movies_master['title']).drop_duplicates()

# ── Save ────────────────────────────────────────────────────────────────────
log.info("Saving artifacts...")
def _save(obj, name):
    with open(f"{MODEL_DIR}/{name}", "wb") as f: pickle.dump(obj, f)

_save(svd,            "svd_movie_model.pkl")
_save(tfidf_genres,   "tfidf_genres.pkl")
_save(tfidf_keywords, "tfidf_keywords.pkl")
_save(tfidf_cast,     "tfidf_cast.pkl")
_save(tfidf_overview, "tfidf_overview.pkl")
_save(indices,        "indices.pkl")
movies_master.to_pickle(f"{MODEL_DIR}/movies_master.pkl")

log.info("All artifacts saved to model/ ✓")
