# Inference API Pseudocode

## Data Models

### Request/Response Models
```pseudocode
Class PromptRequest:
    prompt: string

Class PromptResponse:
    is_correct: integer (0 or 1)
    confidence: float (0.0 to 1.0)
```

## GrammarChecker Class

### Initialization
```pseudocode
Class GrammarChecker:
    Constructor(model_dir = "models"):
        Set device (CPU/GPU)
        Verify model directory exists
        Load tokenizer and model
        Move model to device
        Set model to evaluation mode
```

### Grammar Checking
```pseudocode
Method check_grammar(prompt):
    If prompt is empty:
        Return correct (1) with full confidence

    Tokenize input with:
        - Add special tokens
        - Max length padding
        - Truncation

    Get model prediction:
        - Forward pass
        - Calculate probabilities
        - Get prediction class
        - Get confidence score

    Return prediction and confidence
```

## FastAPI Application

### API Endpoints
```pseudocode
# Health Check
GET /health:
    Return {"status": "healthy"}

# Grammar Check
POST /check:
    Input: PromptRequest
    Output: PromptResponse
    
    Process:
        1. Start timer
        2. Check grammar
        3. Calculate processing time
        4. Log timing
        5. Return result
```

### Error Handling
```pseudocode
For all endpoints:
    Try:
        Process request
    Catch exceptions:
        Log error
        Return 500 with error details
```

## Main Execution Flow
```pseudocode
Initialize FastAPI application
Create GrammarChecker instance
Start uvicorn server:
    - Host: 0.0.0.0
    - Port: 8000
``` 