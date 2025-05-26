# Expert 1.1 Prompt Grammar

This expert system classifies English prompts as having proper or improper grammar and spelling using a fine-tuned DistilBERT model.

## Usage

### 1. Data Preparation
- Place the JFLEG dataset as `data/jfleg.csv` (columns: `prompt`, `label`).
- Run `python src/data.py` to balance and split the data into train/val/test sets.

### 2. Model Training
- Run `python src/model.py` to fine-tune DistilBERT. The best checkpoint is saved in `model_checkpoint/`.

### 3. API
- Start the API with:
  ```
  uvicorn src.api:app --host 127.0.0.1 --port 8000
  ```
- Send POST requests to `/predict` with `{ "prompt": <string> }`.

### 4. Evaluation
- Run `python src/evaluate.py` to evaluate the model on the test set.
- Run `python src/test_api.py` to test the API with example prompts.

### 5. Deployment
- Use the provided `systemd_service_example.txt` to deploy as a systemd service.
- Restrict API access to the bastion host only.

## Notes
- No authentication or rate limiting (internal use only).
- For retraining, repeat steps 1-2 with new data. 