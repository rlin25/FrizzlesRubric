import asyncio
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
import logging
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

# Setup logging
logging.basicConfig(
    filename='/home/ubuntu/orchestrator_api.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

EXPERT_ENDPOINTS = {
    "expert_1_clarity":      "http://172.31.48.199:8008/flow_predict",
    "expert_2_documentation":"http://172.31.48.99:8003/check",
    "expert_3_structure":    "http://172.31.48.22:8004/predict",
    "expert_4_granulation":  "http://172.31.48.104:8005/predict",
    "expert_5_tooling":      "http://172.31.48.12:8006/label",
    "expert_6_repetition":   "http://172.31.48.208:8007/check_prompt"
}
TIMEOUT = 5  # seconds

app = FastAPI()

# Add CORS middleware to allow all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to ["http://127.0.0.1:64452"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

async def call_expert(name, url, prompt):
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(url, json={"prompt": prompt})
            response.raise_for_status()
            data = response.json()
            result = data.get("result", 0)
            if result not in [0, 1]:
                logging.error(f"{name} returned invalid result: {result}")
                return name, 'e'
            logging.info(f"{name} success: {result}")
            return name, result
    except Exception as e:
        logging.error(f"{name} failed: {type(e).__name__}: {e}")
        return name, 'e'

@app.post("/orchestrate")
async def orchestrate(request: PromptRequest):
    prompt = request.prompt
    logging.info(f"Received prompt: {prompt}")
    tasks = [
        call_expert(name, url, prompt)
        for name, url in EXPERT_ENDPOINTS.items()
    ]
    results = await asyncio.gather(*tasks)
    response = {name: result for name, result in results}
    logging.info(f"Response: {response}")
    return response 