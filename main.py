from fastapi import FastAPI
from pydantic import BaseModel
from recommender import recommend

app = FastAPI()

class MovieRequest(BaseModel):
    movie_code: str

@app.post("/recommend")
def get_recommendations(req: MovieRequest):
    recs = recommend(req.movie_code)
    if not recs:
        return {"error": "Movie code not found."}
    return {"recommendations": recs}
