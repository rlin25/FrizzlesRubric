from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests

class PredictRequest(BaseModel):
    input: str

app = FastAPI()

# Change this to the actual internal address/port of the expert instance
EXPERT1_URL = "http://localhost:8000/predict"

@app.post("/expert1/predict")
def proxy_predict(request: PredictRequest):
    resp = requests.post(EXPERT1_URL, json={"input": request.input})
    return resp.json() 