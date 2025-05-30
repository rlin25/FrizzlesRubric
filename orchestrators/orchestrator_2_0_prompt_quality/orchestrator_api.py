import requests
from fastapi import FastAPI
from pydantic import BaseModel
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

EXPERT_1_1_URL = "http://localhost:8002/predict"  # Grammar
EXPERT_1_0_URL = "http://localhost:8001/predict"  # Clarity

class PromptRequest(BaseModel):
    prompt: str

@app.post("/flow_predict")
async def flow_predict(request: PromptRequest):
    logger.info(f"Received prompt: {request.prompt}")
    # Step 1: Grammar check
    try:
        resp_1_1 = requests.post(EXPERT_1_1_URL, json={"prompt": request.prompt}, timeout=10)
        result_1_1 = resp_1_1.json().get("result")
        logger.info(f"Grammar check result: {result_1_1}")
    except Exception as e:
        logger.error(f"Error contacting expert_1_1: {e}")
        return {"error": f"Error contacting expert_1_1: {e}"}
    if result_1_1 == 0:
        logger.info("Prompt failed grammar check. Returning 0.")
        return {"result": 0}
    # Step 2: Clarity check
    try:
        resp_1_0 = requests.post(EXPERT_1_0_URL, json={"prompt": request.prompt}, timeout=10)
        resp_1_0_json = resp_1_0.json()
        logger.info(f"Clarity API full response: {resp_1_0_json}")
        result_1_0 = resp_1_0_json.get("predicted_class")
    except Exception as e:
        logger.error(f"Error contacting expert_1_0: {e}")
        return {"error": f"Error contacting expert_1_0: {e}"}
    if result_1_0 is None or result_1_0 == 0:
        logger.info("Prompt failed clarity check (missing or zero predicted_class). Returning 0.")
        return {"result": 0}
    # Passed both checks
    logger.info("Prompt passed both checks. Returning 1.")
    return {"result": 1}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("orchestrator_2_0_prompt_quality.orchestrator_api:app", host="0.0.0.0", port=8008, reload=False) 