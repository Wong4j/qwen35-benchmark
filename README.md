# Qwen3.5-35B-A3B Benchmark

Performance benchmark results for Qwen3.5-35B-A3B on NVIDIA B200 (single GPU).

## Environment

- **GPU**: NVIDIA B200 (183 GB) x 1
- **TRT-LLM**: v1.3.0rc9 on branch `taylor/taylor_qen3.5_perf` (PR-12265)
- **Model**: Qwen3.5-35B-A3B (35B MoE, 256 experts, top-8)
- **Benchmark tool**: aiperf (synthetic prompts, ignore_eos=true, request_count=8x concurrency)

## Configs

| Config | Backend | Description |
|--------|---------|-------------|
| `qwen3.5_moe_35b_tp1_taylor.yaml` | AutoDeploy | Taylor's optimized config, max_seq_len=262144, max_num_tokens=16000, fine-grained batch sizes 1~32 |
| `qwen3.5_moe_35b_tp1_taylor_c40.yaml` | AutoDeploy | Same as above but with batch sizes 1~48 for c=40 optimization |
| `taylor-lee-20260320-original.yaml` | AutoDeploy | Original 8-GPU 122B config from Taylor (reference only) |

## Results

### AutoDeploy

#### ISL=1k/OSL=100 (Taylor config)

| Concurrency | Output TPS | RPS |
|------------|-----------|-----|
| 8 | 908.5 | 9.08 |
| 16 | 1,454.5 | 14.55 |
| 32 | 2,031.1 | 20.31 |
| 64 | 2,598.8 | 25.99 |
| 128 | 3,342.6 | 33.43 |
| 256 | 3,717.0 | 37.17 |

#### ISL=10k/OSL=350 c=40

| Config | Output TPS | RPS | Notes |
|--------|-----------|-----|-------|
| Taylor standard (graph: 1~32, 64, 128, 256) | 1,204.8 | 3.44 | |
| Taylor c40-optimized (graph: 1~48, 64, 128, 256) | 1,263.9 | 3.61 | +4.9% from finer batch sizes |

### PyTorch Backend

_TODO_

## Usage

### Start server
```bash
./scripts/start_server.sh configs/qwen3.5_moe_35b_tp1_taylor.yaml 8088
```

### Run benchmark
```bash
# ISL=1k/OSL=100
./scripts/run_benchmark.sh configs/qwen3.5_moe_35b_tp1_taylor.yaml 8088 isl1k_osl100

# ISL=10k/OSL=350
./scripts/run_benchmark.sh configs/qwen3.5_moe_35b_tp1_taylor_c40.yaml 8088 isl10k_osl350

# ISL=4k/OSL=1k
./scripts/run_benchmark.sh configs/qwen3.5_moe_35b_tp1_taylor.yaml 8088 isl4k_osl1k
```

### Extract results
```bash
python3 scripts/extract_results.py results/
```
