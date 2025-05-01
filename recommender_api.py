from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from recommender_module import getRecommendations, printRecommendations

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/recommend")
def recommend(title: str = Query(...), top_k: int = Query(10)):
    return {"results": printRecommendations(getRecommendations(title))}
