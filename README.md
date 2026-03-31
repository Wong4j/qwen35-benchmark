# Qwen3.5-35B-A3B AutoDeploy Benchmark

Benchmark results for TensorRT-LLM AutoDeploy backend with Qwen3.5-35B-A3B on NVIDIA B200 (single GPU).

## Environment

- **GPU**: NVIDIA B200 (183 GB) x 1
- **TRT-LLM**: v1.3.0rc9 on branch `taylor/taylor_qen3.5_perf` (PR-12265)
- **Model**: Qwen3.5-35B-A3B (35B MoE, 256 experts, top-8)
- **Backend**: AutoDeploy (`_autodeploy`)
- **Benchmark tool**: aiperf (synthetic prompts, ignore_eos=true, request_count=8x concurrency)

## Configs

| Config | Description |
|--------|-------------|
| `qwen3.5_moe_35b_tp1.yaml` | Basic tp1 config, max_seq_len=65536 |
| `qwen3.5_moe_35b_tp1_taylor.yaml` | Taylor's optimized config, max_seq_len=262144, max_num_tokens=16000, fine-grained batch sizes 1~32 |
| `qwen3.5_moe_35b_tp1_taylor_c40.yaml` | Same as taylor but with batch sizes 1~48 for c=40 optimization |
| `taylor-lee-20260320-original.yaml` | Original 8-GPU 122B config from Taylor (reference only) |

## Results Summary

### ISL=1k/OSL=100 (Taylor config)

| Concurrency | Output TPS | RPS |
|------------|-----------|-----|
| 8 | 915.6 | 9.16 |
| 16 | 1,450.5 | 14.50 |
| 32 | 2,031.0 | 20.31 |
| 64 | 2,631.1 | 26.31 |
| 128 | 3,266.0 | 32.66 |
| 256 | 3,812.9 | 38.13 |

### ISL=10k/OSL=250 c=40

| Config | Output TPS | RPS | Notes |
|--------|-----------|-----|-------|
| Taylor standard (graph: 1~32, 64, 128, 256) | 987.4 | 3.95 | |
| Taylor c40-optimized (graph: 1~48, 64, 128, 256) | 1,018.8 | 4.08 | +3.2% from finer batch sizes |

### ISL=4k/OSL=1k (tp1 config, max_seq_len=65536)

| Concurrency | AutoDeploy TPS | SGLang TPS (ref) | PyT TPS (ref) | AD vs SGLang |
|------------|---------------|-----------------|---------------|--------------|
| 1 | 210.0 | 195 | 211 | 1.08x |
| 8 | 971.4 | 1,153 | 1,191 | 0.84x |
| 64 | 2,977.1 | 4,060 | 4,308 | 0.73x |
| 128 | 3,617.5 | 5,325 | 5,632 | 0.68x |

> SGLang and PyTorch backend reference numbers from colleague's testing on same hardware.

## Usage

### Start server
```bash
./scripts/start_server.sh configs/qwen3.5_moe_35b_tp1_taylor.yaml 8088
```

### Run benchmark
```bash
# ISL=1k/OSL=100
./scripts/run_benchmark.sh configs/qwen3.5_moe_35b_tp1_taylor.yaml 8088 isl1k_osl100

# ISL=10k/OSL=250
./scripts/run_benchmark.sh configs/qwen3.5_moe_35b_tp1_taylor_c40.yaml 8088 isl10k_osl250

# ISL=4k/OSL=1k
./scripts/run_benchmark.sh configs/qwen3.5_moe_35b_tp1_taylor.yaml 8088 isl4k_osl1k
```

### Extract results
```bash
python3 scripts/extract_results.py results/
```

## Key Findings

1. **PR-12265 significantly improves over PR#11581** for Shopee's ISL=1k/OSL=100 scenario (+14%~67%)
2. **ISL=4k/OSL=1k**: AutoDeploy matches SGLang at c=1, but falls behind at high concurrency (16%~32% gap)
3. **PyTorch backend outperforms AutoDeploy** across all scenarios, indicating optimization headroom
4. **Fine-tuning cuda_graph_batch_sizes** for target concurrency yields ~3% improvement
5. **Taylor config** (max_num_tokens=16000, fine batch sizes) improves over basic tp1 config by 2%~16%
