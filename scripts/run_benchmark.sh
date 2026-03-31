#!/bin/bash
# Qwen3.5-35B-A3B Benchmark Script
# Usage: ./run_benchmark.sh <config_yaml> <port> <scenario>
# scenario: isl1k_osl100 | isl10k_osl350 | isl4k_osl1k

set -e

CONFIG=${1:?"Usage: $0 <config.yaml> <port> <scenario>"}
PORT=${2:-8088}
SCENARIO=${3:-isl1k_osl100}
MODEL_PATH=${MODEL_PATH:-/workspace2/model/Qwen3.5-35B-A3B}
RESULT_DIR=${RESULT_DIR:-./results}

# Scenario definitions
case $SCENARIO in
  isl1k_osl100)
    ISL=1000; OSL=100; MIN_TOKENS=100; TIMEOUT=600
    CONCURRENCIES="8 16 32 64 128 256"
    ;;
  isl10k_osl350)
    ISL=10000; OSL=350; MIN_TOKENS=350; TIMEOUT=1800
    CONCURRENCIES="40"
    ;;
  isl4k_osl1k)
    ISL=4000; OSL=1000; MIN_TOKENS=1000; TIMEOUT=1800
    CONCURRENCIES="1 8 64 128 256"
    ;;
  *)
    echo "Unknown scenario: $SCENARIO"
    echo "Available: isl1k_osl100, isl10k_osl350, isl4k_osl1k"
    exit 1
    ;;
esac

CONFIG_NAME=$(basename "$CONFIG" .yaml)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Benchmark ==="
echo "Config: $CONFIG"
echo "Scenario: $SCENARIO (ISL=$ISL, OSL=$OSL)"
echo "Port: $PORT"
echo "Concurrencies: $CONCURRENCIES"
echo ""

for conc in $CONCURRENCIES; do
  req=$((conc * 8))
  outdir="${RESULT_DIR}/${CONFIG_NAME}_${SCENARIO}_c${conc}_${TIMESTAMP}"
  echo "--- Concurrency=$conc, requests=$req ---"
  aiperf profile \
    --model "$MODEL_PATH" \
    --url "127.0.0.1:${PORT}" \
    --endpoint-type chat --streaming \
    --concurrency "$conc" --request-count "$req" \
    --isl "$ISL" --osl "$OSL" \
    --artifact-dir "$outdir" \
    --num-warmup-requests 1 \
    --extra-inputs "{\"ignore_eos\": true, \"min_tokens\": ${MIN_TOKENS}}" \
    --request-timeout-seconds "$TIMEOUT"

  # Extract key metrics
  python3 -c "
import json
d = json.load(open('${outdir}/profile_export_aiperf.json'))
print(f'  TPS={d[\"output_token_throughput\"][\"avg\"]:.1f}, RPS={d[\"request_throughput\"][\"avg\"]:.2f}')
" 2>/dev/null || echo "  Failed to extract metrics"
  echo ""
done

echo "=== Done ==="
