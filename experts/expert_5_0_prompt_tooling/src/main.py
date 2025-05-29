from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from src.detector import detect_ai_tooling

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/label")
async def label_prompt(request: PromptRequest):
    if not request.prompt or not isinstance(request.prompt, str):
        raise HTTPException(status_code=400, detail="Missing or invalid 'prompt' field.")
    label = detect_ai_tooling(request.prompt)
    return {"result": label}

@app.get("/health")
async def health():
    return {"status": "ok"} 