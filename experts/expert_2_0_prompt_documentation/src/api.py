from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.binary_classifier import DocumentationClassifierTrainer
import torch
import os

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

# Load model at startup
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/run_20250518_070518/best_model.pt'))
trainer = DocumentationClassifierTrainer()
trainer.load(MODEL_PATH)

THRESHOLD = 0.75

@app.post("/check")
def check_documentation(req: PromptRequest):
    try:
        probability = trainer.predict(req.prompt)
        prediction = int(probability > THRESHOLD)
        return {"result": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8003, reload=True) 