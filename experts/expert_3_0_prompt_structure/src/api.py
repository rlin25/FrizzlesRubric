from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import DistilBertFileClassifier

app = FastAPI()

class PredictRequest(BaseModel):
    prompt: str

class PredictResponse(BaseModel):
    label: int
    confidence: float

model = None

def load_model(model_path, device='cpu'):
    global model
    model = DistilBertFileClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

@app.on_event('startup')
def startup_event():
    load_model('models/best_model.pt')  # Adjust path as needed

@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    label, confidence = model.predict(req.prompt)
    return PredictResponse(label=label, confidence=confidence) 