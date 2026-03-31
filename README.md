# Qwen3.5 MoE 性能测试

Qwen3.5 MoE 系列模型在 NVIDIA B200 上的 serving throughput 测试，对比 PyTorch backend 和 AutoDeploy backend。

## 环境

- **GPU**: NVIDIA B200 (183 GB)
- **Benchmark tool**: aiperf 0.6.0 (synthetic prompts, ignore_eos=true, request_count=8x concurrency)

| | AutoDeploy | PyTorch |
|--|-----------|---------|
| **Branch** | `taylor/taylor_qen3.5_perf` (PR-12265) | [`qwen3next-3_5-pyt-perf`](https://github.com/Wong4j/TensorRT-LLM/tree/qwen3next-3_5-pyt-perf) |
| **TRT-LLM** | v1.3.0rc9 | v1.3.0rc9 |

## 配置文件

### Qwen3.5-35B-A3B (TP1, 1x B200)

| Config | Backend | Description |
|--------|---------|-------------|
| `qwen3.5_moe_35b_tp1_taylor.yaml` | AutoDeploy | Taylor 优化配置，CUDA graph batch sizes 1~32 |
| `qwen3.5_moe_35b_tp1_taylor_c40.yaml` | AutoDeploy | 同上，batch sizes 扩展到 1~48 以优化 c=40 场景 |
| `qwen3.5_moe_35b_tp1_pytorch.yaml` | PyTorch | TRTLLM MoE backend，参数与 AutoDeploy 对齐 |
| `qwen3.5_moe_35b_tp1_pytorch_c40.yaml` | PyTorch | 同上，batch sizes 扩展到 1~48 以优化 c=40 场景 |

### Qwen3.5-122B-A10B (TP8, 8x B200)

| Config | Backend | Description |
|--------|---------|-------------|
| `qwen3.5_moe_122b_tp8_autodeploy.yaml` | AutoDeploy | TP8 配置，max_batch_size=128，free_gpu=0.9 |
| `qwen3.5_moe_122b_tp8_pytorch.yaml` | PyTorch | TP8 配置，参数与 AutoDeploy 对齐 |

> 122B 与 35B 配置差异：TP8（8 卡）、max_batch_size 降到 128（SSM cache 更大）、free_gpu_memory_fraction 升到 0.9（8 卡显存充裕）。如遇 OOM 可进一步降低 max_batch_size 或 max_num_tokens。

---

## 测试结果

### Qwen3.5-35B-A3B (TP1, 1x B200)

#### ISL=1k/OSL=100

| Concurrency | AutoDeploy TPS | PyTorch TPS | PyT vs AD |
|:-----------:|:--------------:|:-----------:|:---------:|
| 8 | 908.5 | 941.9 | 1.04x |
| 16 | 1,454.5 | 1,568.4 | 1.08x |
| 32 | 2,031.1 | 2,074.8 | 1.02x |
| 64 | 2,598.8 | 3,157.6 | **1.22x** |
| 128 | 3,342.6 | 4,127.6 | **1.23x** |
| 256 | 3,717.0 | 4,823.2 | **1.30x** |

| Concurrency | AutoDeploy RPS | PyTorch RPS |
|:-----------:|:--------------:|:-----------:|
| 8 | 9.08 | 9.42 |
| 16 | 14.55 | 15.75 |
| 32 | 20.31 | 20.92 |
| 64 | 25.99 | 31.96 |
| 128 | 33.43 | 41.59 |
| 256 | 37.17 | 48.63 |

#### ISL=10k/OSL=350 c=40

| Config | AutoDeploy TPS | AutoDeploy RPS | PyTorch TPS | PyTorch RPS | PyT vs AD |
|--------|:--------------:|:--------------:|:-----------:|:-----------:|:---------:|
| Standard (graph: 1~32, 64, 128, 256) | 1,204.8 | 3.44 | 1,522.9 | 4.35 | **1.26x** |
| c40-optimized (graph: 1~48, 64, 128, 256) | 1,263.9 | 3.61 | 1,622.4 | 4.64 | **1.28x** |

#### 结论

- **ISL=1k/OSL=100**: PyTorch backend 比 AutoDeploy 快 1.02x~1.30x，并发越高优势越大。
- **ISL=10k/OSL=350**: PyTorch 快 1.26~1.28x，标准和 c40 优化配置下均保持优势。
- **c40 优化**: 扩展 batch sizes 到 33~48 在 c=40 场景下为 AutoDeploy 带来 +4.9%、PyTorch 带来 +6.5% 的提升。

### Qwen3.5-122B-A10B (TP8, 8x B200)

> TODO: 待测试

---

## 使用方法

### 启动服务

```bash
# === 35B (TP1, 单卡) ===
MODEL_PATH=/path/to/Qwen3.5-35B-A3B

# AutoDeploy
./scripts/start_server.sh configs/qwen3.5_moe_35b_tp1_taylor.yaml 8088

# PyTorch
./scripts/start_server.sh configs/qwen3.5_moe_35b_tp1_pytorch.yaml 8088 pytorch

# === 122B (TP8, 8卡) ===
MODEL_PATH=/path/to/Qwen3.5-122B-A10B

# AutoDeploy
./scripts/start_server.sh configs/qwen3.5_moe_122b_tp8_autodeploy.yaml 8088

# PyTorch
./scripts/start_server.sh configs/qwen3.5_moe_122b_tp8_pytorch.yaml 8088 pytorch
```

### 运行测试

```bash
# ISL=1k/OSL=100
./scripts/run_benchmark.sh configs/<config>.yaml 8088 isl1k_osl100

# ISL=10k/OSL=350
./scripts/run_benchmark.sh configs/<config>.yaml 8088 isl10k_osl350
```

### 提取结果

```bash
python3 scripts/extract_results.py results/
```
