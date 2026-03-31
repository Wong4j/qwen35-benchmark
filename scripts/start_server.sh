#!/bin/bash
# Start AutoDeploy server
# Usage: ./start_server.sh <config_yaml> [port]

set -e

CONFIG=${1:?"Usage: $0 <config.yaml> [port]"}
PORT=${2:-8088}
MODEL_PATH=${MODEL_PATH:-/workspace2/model/Qwen3.5-35B-A3B}

echo "Starting AutoDeploy server..."
echo "  Model: $MODEL_PATH"
echo "  Config: $CONFIG"
echo "  Port: $PORT"

trtllm-serve "$MODEL_PATH" \
  --host 0.0.0.0 --port "$PORT" \
  --backend _autodeploy \
  --trust_remote_code \
  --extra_llm_api_options "$CONFIG"
