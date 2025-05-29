from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

# Define the request body
class PromptRequest(BaseModel):
    prompt: str

# Define the response body
class PredictionResponse(BaseModel):
    predicted_class: int
    confidence: float

app = FastAPI()

# Model and tokenizer paths
MODEL_PATH = '/home/ubuntu/FrizzlesRubric/experts/expert_1_0_prompt_clarity/models/prompt_clarity_model'
TOKENIZER_NAME = 'distilbert-base-uncased'
DEVICE = 'cpu'  # Change to 'cuda' if running on GPU

# Load model and tokenizer at startup
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

@app.post('/predict', response_model=PredictionResponse)
def predict_clarity(request: PromptRequest):
    text = request.prompt
    if not text or not isinstance(text, str):
        raise HTTPException(status_code=400, detail='Prompt must be a non-empty string.')
    
    # Tokenize input
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).cpu().item()
        confidence = probs[0][predicted_class].cpu().item()
        print(f"[API LOG] Input: {text}")
        print(f"[API LOG] Logits: {logits}")
        print(f"[API LOG] Predicted Class: {predicted_class}")
        print(f"[API LOG] Confidence: {confidence}")

    return PredictionResponse(predicted_class=predicted_class, confidence=confidence) 