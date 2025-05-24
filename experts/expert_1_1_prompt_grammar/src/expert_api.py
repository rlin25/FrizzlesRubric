from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
import os

class PredictRequest(BaseModel):
    input: str

app = FastAPI()

MODEL_DIR = "/home/ubuntu/FrizzlesRubric/experts/expert_1_1_prompt_grammar/models/grammar_model/checkpoint-987"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

@app.post("/predict")
def predict(request: PredictRequest):
    text = request.input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
        prediction = prediction.item()
        confidence = confidence.item()
    return {"prediction": prediction, "confidence": confidence} 