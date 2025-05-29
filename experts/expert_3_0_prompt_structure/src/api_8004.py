from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import DistilBertFileClassifier
import uvicorn

MODEL_PATH = '/home/ubuntu/FrizzlesRubric/experts/expert_3_0_prompt_structure/models/best_model.pt'

def load_model():
    device = 'cpu'
    model = DistilBertFileClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device

app = FastAPI()
model, device = load_model()

class PredictRequest(BaseModel):
    prompt: str

@app.post('/predict')
def predict(req: PredictRequest):
    label, _ = model.predict(req.prompt, device=device)
    return {"result": label}

if __name__ == '__main__':
    uvicorn.run('api_8004:app', host='0.0.0.0', port=8004, reload=False) 