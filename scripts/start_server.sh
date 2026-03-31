#!/bin/bash
# Start TRT-LLM server (AutoDeploy or PyTorch backend)
# Usage: ./start_server.sh <config_yaml> [port] [backend]
# backend: _autodeploy (default) | pytorch

set -e

CONFIG=${1:?"Usage: $0 <config.yaml> [port] [backend]"}
PORT=${2:-8088}
BACKEND=${3:-_autodeploy}
MODEL_PATH=${MODEL_PATH:-/workspace2/model/Qwen3.5-35B-A3B}

echo "Starting TRT-LLM server..."
echo "  Model: $MODEL_PATH"
echo "  Config: $CONFIG"
echo "  Backend: $BACKEND"
echo "  Port: $PORT"

trtllm-serve "$MODEL_PATH" \
  --host 0.0.0.0 --port "$PORT" \
  --backend "$BACKEND" \
  --trust_remote_code \
  --extra_llm_api_options "$CONFIG"
