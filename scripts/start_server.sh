#!/bin/bash
# Start TRT-LLM server (AutoDeploy or PyTorch backend)
# Usage: ./start_server.sh <config_yaml> [port] [backend] [tp_size]
# backend: _autodeploy (default) | pytorch

set -e

CONFIG=${1:?"Usage: $0 <config.yaml> [port] [backend] [tp_size]"}
PORT=${2:-8088}
BACKEND=${3:-_autodeploy}
TP_SIZE=${4:-1}
MODEL_PATH=${MODEL_PATH:?"Set MODEL_PATH env var"}

echo "Starting TRT-LLM server..."
echo "  Model: $MODEL_PATH"
echo "  Config: $CONFIG"
echo "  Backend: $BACKEND"
echo "  TP size: $TP_SIZE"
echo "  Port: $PORT"

trtllm-serve "$MODEL_PATH" \
  --host 0.0.0.0 --port "$PORT" \
  --backend "$BACKEND" \
  --tp_size "$TP_SIZE" \
  --trust_remote_code \
  --extra_llm_api_options "$CONFIG"
