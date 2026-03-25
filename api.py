from fastapi import FastAPI, HTTPException, Query
from recommender import load_artifacts, recommend_cf, recommend_content, recommend_hybrid
from typing import Optional

app = FastAPI(title="Movie Recommender API")

# Load once
arts = load_artifacts()

@app.get("/")
def home():
    return {"message": "API is running 🚀"}

@app.get("/recommend")
def recommend(
    user_id: Optional[int] = Query(None),
    movie_title: Optional[str] = Query(None),
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
            raise HTTPException(status_code=400, detail="Provide user_id or movie_title")

        # 🔥 FIX: return directly (no ["title"])
        return {
            "mode": mode,
            "count": len(recs),
            "recommendations": recs
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))