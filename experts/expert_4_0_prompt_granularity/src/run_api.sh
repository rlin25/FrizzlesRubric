#!/bin/bash
# Sir, this script activates the venv and runs the API on port 8005 from the project root

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

source experts/expert_4_0_prompt_granularity/venv/bin/activate

exec uvicorn experts.expert_4_0_prompt_granularity.src.api:app --host 0.0.0.0 --port 8005 