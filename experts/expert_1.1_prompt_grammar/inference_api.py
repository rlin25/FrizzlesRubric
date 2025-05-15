from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import logging
from pathlib import Path
import time
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Grammar Checker API",
    description="API for checking grammatical correctness of prompts",
    version="1.0.0"
)

class PromptRequest(BaseModel):
    prompt: str

class PromptResponse(BaseModel):
    is_correct: int
    confidence: float

class GrammarChecker:
    def __init__(self, model_dir: str = "models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir)
        
        if not self.model_dir.exists():
            raise ValueError(f"Model directory {model_dir} does not exist")
        
        logger.info("Loading model and tokenizer...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully")

    def check_grammar(self, prompt: str) -> Dict[str, Any]:
        """Check the grammatical correctness of a prompt."""
        if not prompt.strip():
            return {"is_correct": 1, "confidence": 1.0}

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()

        return {
            "is_correct": prediction,
            "confidence": confidence
        }

# Initialize the grammar checker
try:
    checker = GrammarChecker()
except Exception as e:
    logger.error(f"Failed to initialize grammar checker: {e}")
    raise

@app.post("/check", response_model=PromptResponse)
async def check_grammar(request: PromptRequest) -> Dict[str, Any]:
    """Check the grammatical correctness of a prompt."""
    try:
        start_time = time.time()
        result = checker.check_grammar(request.prompt)
        processing_time = time.time() - start_time
        
        logger.info(f"Processed prompt in {processing_time:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 