# app.py

import streamlit as st
from recommender import load_artifacts, recommend_cf, recommend_content, recommend_hybrid

# ── Page config ──
st.set_page_config(page_title="CineMatch 🎬", layout="wide")

# ── UI Styling ──
st.markdown("""
<style>
body { background-color: #0e1117; }
.title { font-size: 42px; font-weight: bold; color: #ff4b4b; margin-bottom: 10px; }
.subtitle { color: #aaa; margin-bottom: 20px; }
.card { background: #1c1f26; padding: 16px; border-radius: 12px; margin-bottom: 12px; border: 1px solid #2a2f3a; transition: 0.2s; }
.card:hover { transform: scale(1.01); border-color: #ff4b4b; }
.movie-title { font-size: 18px; font-weight: bold; color: #ffffff; }
.rank { color: #ff4b4b; font-weight: bold; margin-right: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("<div class='title'>🎬 CineMatch</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered Movie Recommendation System</div>", unsafe_allow_html=True)

# ── Load models & movies ──
@st.cache_resource
def load_arts():
    return load_artifacts()

arts = load_arts()
movies = sorted(arts['movies_master']['title'].dropna().unique())

# ── Display results ──
def show_results(results):
    if not results:
        st.warning("No recommendations found")
        return

    st.success(f"Top {len(results)} Recommendations 🎯")
    for i, title in enumerate(results, 1):
        st.markdown(f"""
        <div class="card">
            <span class="rank">#{i}</span>
            <span class="movie-title">{title}</span>
        </div>
        """, unsafe_allow_html=True)

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(["🔀 Hybrid", "📄 Content", "🤝 Collaborative"])

# ───────── HYBRID ─────────
with tab1:
    st.subheader("🔀 Hybrid Recommendation")
    movie = st.selectbox("Select Movie", movies, key="hybrid_movie")
    user = st.number_input("User ID", 1, 1000, 1, key="hybrid_user")
    n = st.slider("Top N", 1, 10, 5, key="hybrid_top_n")

    if st.button("🚀 Recommend (Hybrid)", key="hybrid_btn"):
        with st.spinner("Finding best matches..."):
            res = recommend_hybrid(user, movie, n, arts)
            show_results(res)

# ───────── CONTENT ─────────
with tab2:
    st.subheader("📄 Content-Based Recommendation")
    movie = st.selectbox("Select Movie", movies, key="content_movie")
    n = st.slider("Top N", 1, 10, 5, key="content_top_n")

    if st.button("🚀 Recommend (Content)", key="content_btn"):
        with st.spinner("Analyzing movie..."):
            res = recommend_content(movie, n, arts)
            show_results(res)

# ───────── CF ─────────
with tab3:
    st.subheader("🤝 Collaborative Filtering")
    user = st.number_input("User ID", 1, 1000, 1, key="cf_user")
    n = st.slider("Top N", 1, 10, 5, key="cf_top_n")

    if st.button("🚀 Recommend (CF)", key="cf_btn"):
        with st.spinner("Computing personalized results..."):
            res = recommend_cf(user, n, arts)
            show_results(res)
