from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from recommender_module import getRecommendations, printRecommendations, get_popular_recommendations

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/recommend")
def recommend(title: str = Query(...), top_k: int = Query(10)):
    try:
        movie_ids = getRecommendations(title)
        results = printRecommendations(movie_ids)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/popular")
def popular():
    try:
        movie_ids = get_popular_recommendations()
        results = printRecommendations(movie_ids)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))