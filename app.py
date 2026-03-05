# app.py — CineMatch: Movie Recommendation System Dashboard
# ─────────────────────────────────────────────────────────────────────────────

import json, os
import streamlit as st
from recommender import load_artifacts, recommend_content, recommend_cf, recommend_hybrid

TMDB_IMG = "https://image.tmdb.org/t/p/w300"

st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cinematic dark UI ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0f !important;
    color: #e8e0d0;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0a1a 0%, #0a0a0f 100%) !important;
    border-right: 1px solid #2a1a3a;
}
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 52px;
    font-weight: 900;
    background: linear-gradient(135deg, #f5c518 0%, #ff6b35 50%, #e91e8c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 6px;
}
.hero-sub {
    color: #8a7a6a;
    font-size: 15px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 300;
}
.movie-card {
    background: linear-gradient(135deg, #12101a 0%, #1a1528 100%);
    border: 1px solid #2a2040;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 14px;
    transition: all 0.2s;
    position: relative;
    overflow: hidden;
}
.movie-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #f5c518, #ff6b35, #e91e8c);
}
.movie-title {
    font-family: 'Playfair Display', serif;
    font-size: 18px;
    font-weight: 700;
    color: #f0e8d8;
    margin-bottom: 4px;
}
.movie-meta {
    font-size: 12px;
    color: #6a5a8a;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}
.movie-overview {
    font-size: 13px;
    color: #9a8a7a;
    line-height: 1.5;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.score-badge {
    display: inline-block;
    background: linear-gradient(135deg, #f5c518, #ff9500);
    color: #000;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 6px;
}
.rating-badge {
    display: inline-block;
    background: #1a2a1a;
    color: #4caf50;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid #2a4a2a;
}
.genre-tag {
    display: inline-block;
    background: #1a1030;
    color: #9a7aaa;
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 10px;
    border: 1px solid #2a1a4a;
    margin-right: 4px;
    margin-bottom: 6px;
}
.stat-box {
    background: linear-gradient(135deg, #12101a, #1a1528);
    border: 1px solid #2a2040;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    font-weight: 700;
    color: #f5c518;
}
.stat-label {
    font-size: 11px;
    color: #6a5a8a;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.stButton>button {
    background: linear-gradient(135deg, #8b1a6b, #c0392b) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
    padding: 12px !important;
    font-size: 14px !important;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #a0216f, #d44530) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(192,57,43,0.3) !important;
}
[data-testid="stSelectbox"] > div,
[data-testid="stNumberInput"] input {
    background: #12101a !important;
    border: 1px solid #2a2040 !important;
    color: #e8e0d0 !important;
    border-radius: 8px !important;
}
.stTabs [data-baseweb="tab-list"] { background: #0f0a1a; border-bottom: 1px solid #2a1a3a; }
.stTabs [data-baseweb="tab"]       { color: #6a5a8a; font-family: 'DM Sans'; }
.stTabs [aria-selected="true"]     { color: #f5c518 !important; border-bottom: 2px solid #f5c518 !important; }
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    color: #f0e8d8;
    border-left: 3px solid #f5c518;
    padding-left: 14px;
    margin-bottom: 18px;
}
img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🎬 Loading recommendation engine…")
def get_artifacts():
    return load_artifacts()

@st.cache_data(show_spinner=False)
def get_meta():
    path = "model/meta.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

try:
    arts   = get_artifacts()
    meta   = get_meta()
    LOADED = True
except Exception as e:
    LOADED = False
    load_err = str(e)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='text-align:center; padding: 10px 0 20px;'>"
                "<span style='font-size:40px'>🎬</span><br>"
                "<span style='font-family:Playfair Display,serif; font-size:20px; "
                "color:#f5c518; font-weight:700'>CineMatch</span></div>",
                unsafe_allow_html=True)
    st.markdown("---")

    if LOADED and meta:
        st.markdown("<div class='stat-box'><div class='stat-num'>{:,}</div>"
                    "<div class='stat-label'>Movies</div></div>".format(meta.get('num_movies', 0)),
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='stat-box'><div class='stat-num'>{:,}</div>"
                    "<div class='stat-label'>Ratings</div></div>".format(meta.get('num_ratings', 0)),
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='stat-box'><div class='stat-num'>{:,}</div>"
                    "<div class='stat-label'>Users</div></div>".format(meta.get('num_users', 0)),
                    unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**SVD Model Performance**")
        st.markdown(f"- RMSE: `{meta.get('svd_rmse', '—')}`")
        st.markdown(f"- MAE : `{meta.get('svd_mae', '—')}`")

    st.markdown("---")
    st.markdown("**Algorithms**")
    st.markdown("- 🤝 Collaborative Filtering (SVD)")
    st.markdown("- 📄 Content-Based (TF-IDF)")
    st.markdown("- 🔀 Hybrid (CF + Content)")
    st.markdown("---")
    st.markdown("<div style='color:#3a2a5a; font-size:11px; text-align:center'>"
                "CineMatch v2.0 · 45K Movies</div>", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>CineMatch</div>"
            "<div class='hero-sub'>AI-Powered Movie Recommendation Engine</div>"
            "<div style='color:#3a2a5a; margin:8px 0 24px; font-size:13px'>"
            "Collaborative Filtering · Content-Based · Hybrid · 45,000+ Movies</div>",
            unsafe_allow_html=True)

if not LOADED:
    st.error(f"⚠️ Could not load model: `{load_err}`\n\nRun `python train.py` first.")
    st.stop()

# ── Movie card helper ──────────────────────────────────────────────────────
def render_movie_card(movie: dict, rank: int, show_score=True, score_label="Score"):
    poster_url = f"{TMDB_IMG}{movie['poster_path']}" if movie.get('poster_path') else None
    col_img, col_info = st.columns([1, 4])

    with col_img:
        if poster_url:
            st.image(poster_url, width=90)
        else:
            st.markdown("<div style='width:90px;height:130px;background:#1a1528;"
                        "border-radius:8px;display:flex;align-items:center;"
                        "justify-content:center;color:#3a2a5a;font-size:28px'>🎬</div>",
                        unsafe_allow_html=True)

    with col_info:
        genres_html = " ".join([
            f"<span class='genre-tag'>{g.strip()}</span>"
            for g in str(movie.get('genres', '')).split()[:4]
        ])
        score_html = (f"<span class='score-badge'>{score_label}: {movie['score']}</span>"
                      if show_score else "")
        rating_html = (f"<span class='rating-badge'>⭐ {movie['vote_average']}</span>"
                       if str(movie.get('vote_average', '')) not in ['nan', 'N/A', ''] else "")

        st.markdown(f"""
        <div class='movie-card'>
            <div class='movie-title'>#{rank} {movie['title']}</div>
            <div class='movie-meta'>{movie.get('release_date','')[:4] or '—'} &nbsp;·&nbsp; {score_html} {rating_html}</div>
            <div style='margin-bottom:8px'>{genres_html}</div>
            <div class='movie-overview'>{movie.get('overview','No description available.')}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔀  Hybrid (Recommended)",
    "📄  Content-Based",
    "🤝  Collaborative Filtering",
])

movie_titles = sorted(arts['movies_master']['title'].dropna().unique().tolist())

# ══════════════════════════════════════════════════════════════════════════
# TAB 1: HYBRID
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Hybrid Recommendations</div>",
                unsafe_allow_html=True)
    st.caption("Combines your personal taste (CF) with movie DNA (Content). Best overall results.")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        h_movie = st.selectbox("Select a movie you like", movie_titles, key="h_movie")
    with c2:
        h_user = st.number_input("Your User ID", min_value=1, max_value=100000,
                                  value=1, step=1, key="h_user")
    with c3:
        h_n = st.selectbox("Results", [5, 8, 10], key="h_n")

    if st.button("🔀  Get Hybrid Recommendations", key="h_btn"):
        with st.spinner("Finding your perfect matches…"):
            results = recommend_hybrid(h_user, h_movie, h_n, arts)
        if results:
            st.success(f"Top {len(results)} recommendations for User {h_user} who likes **{h_movie}**")
            for i, m in enumerate(results, 1):
                render_movie_card(m, i, show_score=True, score_label="Match")
        else:
            st.warning("No results found. Try a different movie.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 2: CONTENT-BASED
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Content-Based Recommendations</div>",
                unsafe_allow_html=True)
    st.caption("Finds movies with similar genres, cast, keywords and plot. No user history needed.")

    c1, c2 = st.columns([3, 1])
    with c1:
        cb_movie = st.selectbox("Select a movie", movie_titles, key="cb_movie")
    with c2:
        cb_n = st.selectbox("Results", [5, 8, 10], key="cb_n")

    if st.button("📄  Find Similar Movies", key="cb_btn"):
        with st.spinner("Analysing movie DNA…"):
            results = recommend_content(cb_movie, cb_n, arts)
        if results:
            st.success(f"Movies similar to **{cb_movie}**")
            for i, m in enumerate(results, 1):
                render_movie_card(m, i, show_score=True, score_label="Similarity")
        else:
            st.warning("Movie not found in database.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 3: COLLABORATIVE FILTERING
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Collaborative Filtering</div>",
                unsafe_allow_html=True)
    st.caption("SVD matrix factorisation — predicts what users with similar taste rated highly.")

    c1, c2 = st.columns([2, 1])
    with c1:
        cf_user = st.number_input("Enter User ID", min_value=1, max_value=100000,
                                   value=5, step=1, key="cf_user")
    with c2:
        cf_n = st.selectbox("Results", [5, 8, 10], key="cf_n")

    st.info("💡 User IDs 1–671 are known users from the training set and will give the most personalised results.")

    if st.button("🤝  Get Personalised Recommendations", key="cf_btn"):
        with st.spinner("Computing personalised scores…"):
            results = recommend_cf(cf_user, cf_n, arts)
        if results:
            st.success(f"Top {len(results)} picks for User {cf_user}")
            for i, m in enumerate(results, 1):
                render_movie_card(m, i, show_score=True, score_label="Predicted Rating")
        else:
            st.warning("No recommendations found for this user.")
