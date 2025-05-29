#!/bin/bash
# Sir's script to start the FastAPI API on port 8006
 
export PYTHONPATH="/home/ubuntu/FrizzlesRubric/experts/expert_5_0_prompt_tooling"
source /home/ubuntu/FrizzlesRubric/experts/expert_5_0_prompt_tooling/venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 8006 