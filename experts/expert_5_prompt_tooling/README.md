# Expert 5 - Tooling

## Running Locally

```powershell
pip install -r requirements.txt
uvicorn src.main:app --reload
```

## Running Tests

```powershell
pytest src/
```

## Docker Usage

Build the image:
```powershell
docker build -t expert5-tooling .
```

Run the container:
```powershell
docker run -p 8000:8000 expert5-tooling
```

## API Endpoints

- `POST /label` — Request body: `{ "prompt": "..." }` → Response: `{ "label": 0 or 1 }`
- `GET /health` — Returns `{ "status": "ok" }` 