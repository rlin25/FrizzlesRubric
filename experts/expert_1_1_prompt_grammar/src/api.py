import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

MODEL_DIR = "/home/ubuntu/FrizzlesRubric/experts/expert_1_1_prompt_grammar/model_checkpoint"
TOKENIZER_NAME = "distilbert-base-uncased"

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/predict")
async def predict(request: PromptRequest):
    prompt = request.prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    return {"result": prediction}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8002, reload=False) 