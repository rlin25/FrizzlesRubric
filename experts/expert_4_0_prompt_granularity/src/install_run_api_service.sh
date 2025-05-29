#!/bin/bash
# Sir, this script installs, enables, and starts the run_api systemd service

SERVICE_SRC="/home/ubuntu/FrizzlesRubric/experts/expert_4_0_prompt_granularity/src/run_api.service"
SERVICE_DEST="/etc/systemd/system/run_api.service"

# Copy the service file
sudo cp "$SERVICE_SRC" "$SERVICE_DEST"

# Reload systemd
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable run_api.service

# Start the service
sudo systemctl start run_api.service

# Show status
sudo systemctl status run_api.service 