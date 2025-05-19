# API Specifications

## Endpoint
```
POST /api/expert3/evaluate
```

## Request Format
```json
{
    "prompt": "string",
    "timestamp": "ISO-8601",
    "options": {
        "timeout_ms": "number",
        "priority": "number",
        "batch_id": "string"
    }
}
```

## Response Format
```json
{
    "score": 0 or 1,
    "processing_time_ms": "number",
    "confidence": "number",
    "metadata": {
        "model_version": "string",
        "pattern_matches": ["string"],
        "processing_steps": ["string"]
    }
}
```

## Rate Limiting
- Requests per second: 100 (increased from 10)
- Burst limit: 200 (increased from 20)
- Per-client limit: 1000 requests/minute
- Global limit: 10000 requests/minute

## Error Handling
### Error Codes
- E001: Invalid input format
- E002: Processing timeout
- E003: Pattern matching error
- E004: ML model error
- E005: Rate limit exceeded
- E006: Resource exhaustion
- E007: Model loading error
- E008: Batch processing error

### Error Response Format
```json
{
    "error": {
        "code": "string",
        "message": "string",
        "details": {
            "timestamp": "ISO-8601",
            "request_id": "string",
            "processing_stage": "string"
        }
    }
}
```

## Performance Requirements
- Response time: < 100ms
- Processing time: < 500ms
- Availability: 99.9%
- Throughput: 100 requests/second
- Concurrent connections: 1000
- Error rate: < 1%

## Security
- Authentication: None required
- CORS: Enabled
- Allowed origins: ["*"]
- Rate limiting: Enabled
- Input validation: Strict
- Output sanitization: Enabled

## Monitoring
### Metrics
- Request count
- Response time
- Error rate
- Processing time
- Resource usage
- Model performance
- Batch statistics

### Logging
- Request details
- Response details
- Error details
- Performance metrics
- Resource usage
- Model metrics
- System health

## Batch Processing
- Batch size: 64
- Batch timeout: 5000ms
- Retry attempts: 3
- Backoff strategy: Exponential
- Error handling: Per-item

## Example Usage
### Single Request
```bash
curl -X POST http://api/expert3/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Add error handling to the fetchData function in src/api/data.js",
    "timestamp": "2024-03-15T12:00:00Z",
    "options": {
        "timeout_ms": 500,
        "priority": 1,
        "batch_id": "batch_123"
    }
  }'
```

### Batch Request
```bash
curl -X POST http://api/expert3/evaluate/batch \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
        {
            "prompt": "Add error handling to the fetchData function in src/api/data.js",
            "timestamp": "2024-03-15T12:00:00Z"
        },
        {
            "prompt": "Fix the memory leak in components/DataTable.js",
            "timestamp": "2024-03-15T12:00:01Z"
        }
    ],
    "options": {
        "timeout_ms": 5000,
        "priority": 1,
        "batch_id": "batch_123"
    }
  }'
```

### Response
```json
{
    "score": 1,
    "processing_time_ms": 150,
    "confidence": 0.95,
    "metadata": {
        "model_version": "1.0.0",
        "pattern_matches": [
            "file_path",
            "function_reference"
        ],
        "processing_steps": [
            "input_validation",
            "pattern_matching",
            "ml_inference"
        ]
    }
}
```

### Error Response
```json
{
    "error": {
        "code": "E001",
        "message": "Invalid input format",
        "details": {
            "timestamp": "2024-03-15T12:00:00Z",
            "request_id": "req_123",
            "processing_stage": "input_validation"
        }
    }
}
``` 