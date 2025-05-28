# Expert 6.0 Prompt Repetition Detection

This project implements a prompt repetition detection system as specified in the masterplan. It includes preprocessing, embedding generation, encryption, DynamoDB storage, an API, logging, analytics, and manual review tools.

## Project Structure

- `src/` - Main source code for the pipeline and API
- `scripts/` - Utility and manual review scripts
- `tests/` - Unit and integration tests
- `docs/` - Documentation and plans
- `venv/` - Python virtual environment

## Setup

1. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- Run the API server:
  ```bash
  python src/api_server.py
  ```
- Run manual review or analytics scripts as needed from `scripts/`.

## AWS Setup
- Ensure you have AWS credentials with access to KMS and DynamoDB.
- DynamoDB tables required: `Prompts`, `PromptChecks`.

## Plans
See `docs/plans/masterplan.md` for full implementation details. 