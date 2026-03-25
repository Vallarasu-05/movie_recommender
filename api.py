from fastapi import FastAPI, HTTPException, Query
from recommender import load_artifacts, recommend_cf, recommend_content, recommend_hybrid

app = FastAPI(title="Movie Recommender API")

# Load once (VERY IMPORTANT)
arts = load_artifacts()

@app.get("/")
def home():
    return {"message": "API is running 🚀"}

@app.get("/recommend")
def recommend(
    user_id: int = Query(None),
    movie_title: str = Query(None),
    top_n: int = 5
):

    try:
        if user_id and movie_title:
            recs = recommend_hybrid(user_id, movie_title, top_n, arts)
            mode = "hybrid"

        elif user_id:
            recs = recommend_cf(user_id, top_n, arts)
            mode = "collaborative"

        elif movie_title:
            recs = recommend_content(movie_title, top_n, arts)
            mode = "content"

        else:
            raise HTTPException(400, "Provide user_id or movie_title")

        return {
            "mode": mode,
            "count": len(recs),
            "recommendations": [r["title"] for r in recs]
        }

    except Exception as e:
        raise HTTPException(500, str(e))