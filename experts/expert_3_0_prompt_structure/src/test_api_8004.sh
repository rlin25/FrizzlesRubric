#!/bin/bash

while true; do
  read -p "Enter prompt (or type 'exit' to quit): " prompt
  if [[ "$prompt" == "exit" ]]; then
    echo "Exiting."
    break
  fi
  # Use jq to encode the prompt as valid JSON
  json=$(jq -n --arg prompt "$prompt" '{prompt: $prompt}')
  response=$(curl -s -X POST http://localhost:8004/predict \
    -H "Content-Type: application/json" \
    -d "$json")
  prediction=$(echo "$response" | jq -r '.prediction')
  echo "$prediction"
done 