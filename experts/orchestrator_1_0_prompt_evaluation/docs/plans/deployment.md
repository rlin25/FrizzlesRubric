# Deployment

- Use FastAPI as the web framework
- Use httpx for async HTTP requests
- Use uvicorn as the ASGI server
- Install dependencies: fastapi, httpx, uvicorn
- Run with: uvicorn orchestrator_api:app --host 0.0.0.0 --port 8010
- Optionally, create a Systemd service for automatic startup 