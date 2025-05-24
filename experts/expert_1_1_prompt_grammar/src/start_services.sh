#!/bin/bash

# Start expert_api service
nohup python3 -m uvicorn expert_api:app --host 0.0.0.0 --port 8002 > expert_api.log 2>&1 &

# Start bastion_api service
nohup python3 -m uvicorn bastion_api:app --host 0.0.0.0 --port 8082 > bastion_api.log 2>&1 &

echo "Both services started. Logs: expert_api.log, bastion_api.log" 