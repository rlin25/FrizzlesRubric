#!/bin/bash

# This script creates systemd service files for expert_api and bastion_api
# Run as root or with sudo

SERVICE_DIR="/etc/systemd/system"
WORKDIR="/home/ubuntu/FrizzlesRubric/experts/expert_1_0_prompt_clarity/src"
VENV_BIN="/home/ubuntu/FrizzlesRubric/experts/expert_1_0_prompt_clarity/venv/bin"
USER="ubuntu"

# expert_api.service
cat <<EOF | sudo tee $SERVICE_DIR/expert_api.service > /dev/null
[Unit]
Description=Expert 1.0 Prompt Clarity API
After=network.target

[Service]
User=$USER
WorkingDirectory=$WORKDIR
ExecStart=$VENV_BIN/python3 -m uvicorn api:app --host 0.0.0.0 --port 8001
Restart=always
StandardOutput=append:$WORKDIR/expert_api.log
StandardError=append:$WORKDIR/expert_api.log
Environment="PATH=$VENV_BIN:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

[Install]
WantedBy=multi-user.target
EOF

# bastion_api.service
cat <<EOF | sudo tee $SERVICE_DIR/bastion_api.service > /dev/null
[Unit]
Description=Expert 1.0 Bastion API
After=network.target

[Service]
User=$USER
WorkingDirectory=$WORKDIR
ExecStart=$VENV_BIN/python3 -m uvicorn bastion_api:app --host 0.0.0.0 --port 8081
Restart=always
StandardOutput=append:$WORKDIR/bastion_api.log
StandardError=append:$WORKDIR/bastion_api.log
Environment="PATH=$VENV_BIN:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

[Install]
WantedBy=multi-user.target
EOF

echo "Systemd service files created. Run:"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable expert_api bastion_api"
echo "  sudo systemctl start expert_api bastion_api" 