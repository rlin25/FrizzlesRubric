#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 'Your input text here'"
  exit 1
fi

INPUT_TEXT="$1"

# This script sends a request to the bastion API, which proxies to the expert API.
# Ensure the expert API is loading the correct trained checkpoint (e.g., checkpoint-987).
# If you want to test a different model, update the backend API, not this script.
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"input": "'$INPUT_TEXT'"}' \
  http://localhost:8080/expert1/predict 