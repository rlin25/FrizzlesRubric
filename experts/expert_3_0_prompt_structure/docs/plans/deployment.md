# Deployment Plan

## Model Export
- Export the trained DistilBERT classifier and tokenizer.

## Serving
- Serve the model using a lightweight API framework (e.g., FastAPI or Flask).
- Ensure the API can handle real-time requests from the web app with low latency.

## Performance Monitoring
- Monitor inference latency and throughput in production.
- Log predictions and errors for future analysis (optional). 