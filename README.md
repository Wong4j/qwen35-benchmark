# Qwen3.5-35B-A3B Benchmark

Performance benchmark results for Qwen3.5-35B-A3B on NVIDIA B200 (single GPU).

## Environment

- **GPU**: NVIDIA B200 (183 GB) x 1
- **Model**: Qwen3.5-35B-A3B BF16 TP1 (35B MoE, 256 experts, top-8)
- **Benchmark tool**: aiperf 0.6.0 (synthetic prompts, ignore_eos=true, request_count=8x concurrency)

| | AutoDeploy | PyTorch |
|--|-----------|---------|
| **Branch** | `taylor/taylor_qen3.5_perf` (PR-12265) | [`qwen3next-3_5-pyt-perf`](https://github.com/Wong4j/TensorRT-LLM/tree/qwen3next-3_5-pyt-perf) |
| **TRT-LLM** | v1.3.0rc9 | v1.3.0rc9 |
| **max_seq_len** | 262144 | 262144 |
| **max_num_tokens** | 16000 | 16000 |
| **max_batch_size** | 256 | 256 |
| **free_gpu_memory_fraction** | 0.4 | 0.4 |

## Configs

| Config | Backend | Description |
|--------|---------|-------------|
| `qwen3.5_moe_35b_tp1_taylor.yaml` | AutoDeploy | Taylor's optimized config, fine-grained batch sizes 1~32 |
| `qwen3.5_moe_35b_tp1_taylor_c40.yaml` | AutoDeploy | Same as above but with batch sizes 1~48 for c=40 optimization |
| `qwen3.5_moe_35b_tp1_pytorch.yaml` | PyTorch | TRTLLM MoE backend, matched parameters with AutoDeploy |
| `qwen3.5_moe_35b_tp1_pytorch_c40.yaml` | PyTorch | Same as above but with batch sizes 1~48 for c=40 optimization |
| `taylor-lee-20260320-original.yaml` | AutoDeploy | Original 8-GPU 122B config from Taylor (reference only) |

## Results

### ISL=1k/OSL=100

| Concurrency | AutoDeploy TPS | PyTorch TPS | PyT vs AD |
|------------|---------------|-------------|-----------|
| 8 | 908.5 | 941.9 | 1.04x |
| 16 | 1,454.5 | 1,568.4 | 1.08x |
| 32 | 2,031.1 | 2,074.8 | 1.02x |
| 64 | 2,598.8 | 3,157.6 | 1.22x |
| 128 | 3,342.6 | 4,127.6 | 1.23x |
| 256 | 3,717.0 | 4,823.2 | 1.30x |

| Concurrency | AutoDeploy RPS | PyTorch RPS |
|------------|---------------|-------------|
| 8 | 9.08 | 9.42 |
| 16 | 14.55 | 15.75 |
| 32 | 20.31 | 20.92 |
| 64 | 25.99 | 31.96 |
| 128 | 33.43 | 41.59 |
| 256 | 37.17 | 48.63 |

### ISL=10k/OSL=350 c=40

| Config | AutoDeploy TPS | AutoDeploy RPS | PyTorch TPS | PyTorch RPS | PyT vs AD |
|--------|---------------|----------------|-------------|-------------|-----------|
| Standard (graph: 1~32, 64, 128, 256) | 1,204.8 | 3.44 | 1,522.9 | 4.35 | 1.26x |
| c40-optimized (graph: 1~48, 64, 128, 256) | 1,263.9 | 3.61 | 1,622.4 | 4.64 | 1.28x |

### Summary

- **ISL=1k/OSL=100**: PyTorch backend is 1.02x~1.30x faster than AutoDeploy. Gap widens at higher concurrency.
- **ISL=10k/OSL=350**: PyTorch is 1.26~1.28x faster across both standard and c40-optimized configs.
- **c40 optimization**: Adding batch sizes 33~48 yields +4.9% for AutoDeploy and +6.5% for PyTorch on c=40 workloads.

## Usage

### Start server
```bash
# AutoDeploy backend
./scripts/start_server.sh configs/qwen3.5_moe_35b_tp1_taylor.yaml 8088

# PyTorch backend
./scripts/start_server.sh configs/qwen3.5_moe_35b_tp1_pytorch.yaml 8088 pytorch
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
