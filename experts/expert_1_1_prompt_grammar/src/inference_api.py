from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import logging
from pathlib import Path
import time
from typing import Dict, Any, Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import asyncio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# API configuration
API_TIMEOUT = 5.0  # seconds
MAX_PROMPT_LENGTH = 1000  # characters

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    try:
        app.state.checker = GrammarChecker()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to initialize grammar checker: {e}")
        raise
    yield
    # Shutdown: Clean up resources
    if hasattr(app.state, 'checker'):
        del app.state.checker

app = FastAPI(
    title="Grammar Checker API",
    description="""
    API for checking grammatical correctness of prompts.
    
    Features:
    - Binary classification of grammatical correctness
    - Confidence score for predictions
    - Rate limiting for API stability
    - Request timeout handling
    - Input validation
    
    Example request:
    ```json
    {
        "prompt": "This is a grammatically correct sentence."
    }
    ```
    
    Example response:
    ```json
    {
        "is_correct": 1,
        "confidence": 0.95
    }
    ```
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiter error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_LENGTH)
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty or contain only whitespace")
        return v

class PromptResponse(BaseModel):
    is_correct: int = Field(..., description="1 for correct, 0 for incorrect")
    confidence: float = Field(..., description="Confidence score between 0 and 1")
    processing_time: float = Field(..., description="Processing time in seconds")

class GrammarChecker:
    def __init__(self, model_dir: str = "models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir)
        
        if not self.model_dir.exists():
            raise ValueError(f"Model directory {model_dir} does not exist")
        
        logger.info("Loading model and tokenizer...")
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=2
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    async def check_grammar(self, prompt: str) -> Dict[str, Any]:
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

@app.post("/check", response_model=PromptResponse)
@limiter.limit("10/minute")
async def check_grammar(
    request: Request,
    prompt_request: PromptRequest
) -> Dict[str, Any]:
    """
    Check the grammatical correctness of a prompt.
    
    Args:
        prompt_request: The prompt to check
        
    Returns:
        Dict containing:
        - is_correct: 1 for correct, 0 for incorrect
        - confidence: Confidence score between 0 and 1
        - processing_time: Processing time in seconds
        
    Raises:
        HTTPException: If processing fails or times out
    """
    try:
        start_time = time.time()
        
        # Process with timeout
        try:
            result = await asyncio.wait_for(
                app.state.checker.check_grammar(prompt_request.prompt),
                timeout=API_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Request timed out. Please try again with a shorter prompt."
            )
        
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
        
        logger.info(
            f"Processed prompt in {processing_time:.2f}s. "
            f"Length: {len(prompt_request.prompt)}, "
            f"Result: {result}"
        )
        
        return result
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for orchestrator monitoring.
    
    Returns:
        Dict containing:
        - status: "healthy" if the service is running properly
    """
    return {"status": "healthy"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=30,
        log_level="info"
    ) 