import streamlit as st
from recommender import load_artifacts, recommend_content, recommend_hybrid

st.set_page_config(page_title="CineMatch 🎬", layout="wide")

# ───────────────── HEADER ─────────────────
st.markdown("<h1 style='color:#ff4b4b;'>🎬 CineMatch</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color:#aaa;'>AI-Powered Movie Recommendation System</h3>", unsafe_allow_html=True)

# ───────────────── LOAD ARTIFACTS ─────────────────
arts = load_artifacts()
movies = sorted(arts['movies_master']['title'].dropna().unique())

# ───────────────── RESULT DISPLAY ─────────────────
def show_results(results):
    if not results:
        st.warning("No recommendations found")
        return

    st.success(f"Top {len(results)} Recommendations 🎯")
    for i, title in enumerate(results, 1):
        st.markdown(f"<div style='background:#1c1f26; padding:8px; margin:4px; border-radius:6px; color:#fff;'><b>#{i}</b> {title}</div>", unsafe_allow_html=True)

# ───────────────── TABS ─────────────────
tab1, tab2 = st.tabs(["📄 Content", "🔀 Hybrid"])

# ───────── CONTENT-BASED ─────────
with tab1:
    st.subheader("📄 Content-Based Recommendation")
    movie = st.selectbox("Select Movie", movies, key="content")
    n = st.slider("Top N", 1, 10, 5, key="content_n")

    if st.button("🚀 Recommend (Content)", key="btn_content"):
        with st.spinner("Finding recommendations..."):
            results = recommend_content(movie, n, arts)
            show_results(results)

# ───────── HYBRID (content-based only) ─────────
with tab2:
    st.subheader("🔀 Hybrid Recommendation")
    movie = st.selectbox("Select Movie", movies, key="hybrid")
    user_id = st.number_input("User ID (ignored, CF removed)", 1, 1000, 1)
    n = st.slider("Top N", 1, 10, 5, key="hybrid_n")

    if st.button("🚀 Recommend (Hybrid)", key="btn_hybrid"):
        with st.spinner("Finding recommendations..."):
            results = recommend_hybrid(user_id, movie, n, arts)
            show_results(results)
