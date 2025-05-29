# INCOMING BRANCH VERSION

# Grammar Checker Expert 1.1

A FastAPI-based service for checking grammatical correctness of prompts.

## Features

- Binary classification of grammatical correctness
- Confidence score for predictions
- Rate limiting (10 requests per minute)
- Request timeout handling (5 seconds)
- Input validation
- Health check endpoint for orchestrator monitoring

## API Endpoints

### POST /check
Check the grammatical correctness of a prompt.

Request:
```json
{
    "prompt": "This is a grammatically correct sentence."
}
```

Response:
```json
{
    "is_correct": 1,
    "confidence": 0.95,
    "processing_time": 0.123
}
```

### GET /health
Health check endpoint for orchestrator monitoring.

Response:
```json
{
    "status": "healthy"
}
```

## Deployment

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the server:
```bash
python inference_api.py
```

The server will start on `http://localhost:8000`.

## API Documentation

Once the server is running, you can access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Error Handling

The API handles various error cases:
- Empty or whitespace-only prompts
- Prompts exceeding maximum length (1000 characters)
- Rate limit exceeded (10 requests per minute)
- Request timeout (5 seconds)
- Server errors

## Monitoring

The service includes:
- Detailed logging with timestamps
- Processing time tracking
- Request/response logging
- Error tracking

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)
- 2GB+ RAM
- 500MB+ disk space for model files 
