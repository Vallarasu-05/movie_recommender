import streamlit as st
import requests
import pickle

API_URL = "http://127.0.0.1:8000/recommend"

st.set_page_config(page_title="CineMatch 🎬", layout="wide")

# ───────────────── UI STYLING ─────────────────
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.title {
    font-size: 42px;
    font-weight: bold;
    color: #ff4b4b;
    margin-bottom: 10px;
}
.subtitle {
    color: #aaa;
    margin-bottom: 20px;
}
.card {
    background: #1c1f26;
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 12px;
    border: 1px solid #2a2f3a;
    transition: 0.2s;
}
.card:hover {
    transform: scale(1.01);
    border-color: #ff4b4b;
}
.movie-title {
    font-size: 18px;
    font-weight: bold;
    color: #ffffff;
}
.rank {
    color: #ff4b4b;
    font-weight: bold;
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)

# ───────────────── HEADER ─────────────────
st.markdown("<div class='title'>🎬 CineMatch</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered Movie Recommendation System</div>", unsafe_allow_html=True)

# ───────────────── LOAD MOVIES ─────────────────
@st.cache_data
def load_movies():
    with open("models/movies_master.pkl", "rb") as f:
        movies = pickle.load(f)
    return sorted(movies['title'].dropna().unique())

movies = load_movies()

# ───────────────── RESULT DISPLAY ─────────────────
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

# ───────────────── TABS ─────────────────
tab1, tab2, tab3 = st.tabs(["🔀 Hybrid", "📄 Content", "🤝 Collaborative"])

# ───────── HYBRID ─────────
with tab1:
    st.subheader("🔀 Hybrid Recommendation")

    movie = st.selectbox("Select Movie", movies)
    user = st.number_input("User ID", 1, 1000, 1)
    n = st.slider("Top N", 1, 10, 5)

    if st.button("🚀 Recommend (Hybrid)"):
        with st.spinner("Finding best matches..."):
            res = requests.get(API_URL, params={
                "user_id": user,
                "movie_title": movie,
                "top_n": n
            })

        if res.status_code == 200:
            data = res.json()
            show_results(data["recommendations"])
        else:
            st.error("Error fetching recommendations")

# ───────── CONTENT ─────────
with tab2:
    st.subheader("📄 Content-Based Recommendation")

    movie = st.selectbox("Select Movie", movies, key="cb")
    n = st.slider("Top N", 1, 10, 5, key="cbn")

    if st.button("🚀 Recommend (Content)"):
        with st.spinner("Analyzing movie..."):
            res = requests.get(API_URL, params={
                "movie_title": movie,
                "top_n": n
            })

        if res.status_code == 200:
            data = res.json()
            show_results(data["recommendations"])
        else:
            st.error("Error fetching recommendations")

# ───────── CF ─────────
with tab3:
    st.subheader("🤝 Collaborative Filtering")

    user = st.number_input("User ID", 1, 1000, 1, key="cf")
    n = st.slider("Top N", 1, 10, 5, key="cfn")

    if st.button("🚀 Recommend (CF)"):
        with st.spinner("Computing personalized results..."):
            res = requests.get(API_URL, params={
                "user_id": user,
                "top_n": n
            })

        if res.status_code == 200:
            data = res.json()
            show_results(data["recommendations"])
        else:
            st.error("Error fetching recommendations")