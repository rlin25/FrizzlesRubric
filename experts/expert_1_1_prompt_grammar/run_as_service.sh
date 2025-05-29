#!/bin/bash

# Absolute paths
VENV_PATH="/home/ubuntu/FrizzlesRubric/experts/expert_1_1_prompt_grammar/venv/bin/python"
API_PATH="/home/ubuntu/FrizzlesRubric/experts/expert_1_1_prompt_grammar/src/api.py"
SERVICE_NAME="prompt_grammar_api"

# Create systemd service file
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null <<EOL
[Unit]
Description=Prompt Grammar FastAPI Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/FrizzlesRubric/experts/expert_1_1_prompt_grammar
ExecStart=${VENV_PATH} ${API_PATH}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOL

# Reload systemd, enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl start ${SERVICE_NAME}

echo "Service ${SERVICE_NAME} started and enabled. Check status with: sudo systemctl status ${SERVICE_NAME}"