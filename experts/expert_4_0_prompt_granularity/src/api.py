from fastapi import FastAPI
from pydantic import BaseModel
import torch
from pathlib import Path
from experts.expert_4_0_prompt_granularity.src.model import GranularityClassifier

app = FastAPI()

MODEL_PATH = "/home/ubuntu/FrizzlesRubric/experts/expert_4_0_prompt_granularity/models/checkpoints/best_model.pt"
THRESHOLD = 0.6

class PromptRequest(BaseModel):
    prompt: str

# Load model at startup
model = GranularityClassifier()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

@app.post("/predict")
async def predict(request: PromptRequest):
    pred, _ = model.predict(request.prompt, device=device)
    result = 1 if pred > THRESHOLD else 0
    return {"result": result} 