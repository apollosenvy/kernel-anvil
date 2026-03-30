# kernel-anvil

Profile-guided GPU kernel optimizer for AMD. Reads a GGUF model, profiles each layer's GEMV shape on your GPU, and generates optimal kernel configs that llama.cpp loads at runtime. No recompilation needed.

**2x decode speedup on Qwen3.5-27B** (12 -> 27 tok/s on a 7900 XTX) from shape-specific kernel tuning alone.

## How It Works

llama.cpp's quantized GEMV kernels (MMVQ) use one-size-fits-all parameters for all layer shapes. But optimal configs vary dramatically by shape -- a 1024-row GQA projection wants different thread/block settings than a 17408-row FFN layer.

kernel-anvil profiles each unique (quant_type, N, K) shape from your model on your actual GPU, finds the fastest config via guided sweep, and writes a JSON config file that llama.cpp reads at startup.

```
kernel-anvil gguf-optimize model.gguf    # profiles 8 shapes in <1s
llama-server -m model.gguf               # auto-loads optimized configs
```

## Install

```bash
git clone https://github.com/apollosenvy/kernel-anvil.git
cd kernel-anvil && pip install -e ".[dev]"
```

Requires: Python 3.10+, PyTorch 2.0+ (with ROCm), Triton 3.0+

> **ROCm Triton note:** If `pip install` fails on the Triton dependency, install the ROCm-compatible Triton wheel from [PyTorch's index](https://download.pytorch.org/whl/rocm7.1/): `pip install triton --index-url https://download.pytorch.org/whl/rocm7.1/`

## Usage

### Optimize a model (one time)

```bash
kernel-anvil gguf-optimize ~/Models/Qwen3-8B-Q4_K_M.gguf
```

Profiles every unique GEMV shape, writes optimal configs to `~/.cache/smithy/<model>.json`.

### Run llama.cpp with optimized configs

```bash
SMITHY_CONFIG=~/.cache/smithy/Qwen3-8B-Q4_K_M.json \
    llama-server -m ~/Models/Qwen3-8B-Q4_K_M.gguf -ngl 999
```

On startup:
```
smithy: loaded 6 shape-specific kernel configs from ~/.cache/smithy/Qwen3-8B-Q4_K_M.json
```

### Profile a Triton kernel

```bash
kernel-anvil sweep examples/simple_gemv.py
kernel-anvil profile examples/simple_gemv.py
```

## Results

### Qwen3.5-27B-Claude-Distill Q4_K_M on 7900 XTX

| Shape | Count | Speedup | Optimization |
|-------|------:|--------:|-------------|
| Q4_K 5120x6144 | 48 | **1.54x** | rows_per_block=2 |
| Q5_K 5120x10240 | 48 | **1.38x** | nwarps=8 |
| Q4_K 6144x5120 | 64 | **1.17x** | nwarps=8 |
| Q4_K 5120x1024 | 22 | **1.13x** | rows_per_block=2 |

**End-to-end: 12 tok/s -> 27 tok/s decode (2.25x)**

> **Note:** The 12 tok/s baseline was measured with [TurboQuant](https://github.com/TheTom/turboquant_plus) turbo3 KV cache compression enabled, which adds decode overhead. Stock llama.cpp without TurboQuant gets ~20 tok/s on this model. The 2.25x improvement is the combined effect of kernel-anvil's shape-specific tuning on top of TurboQuant's compressed KV path.

### Qwen3-8B Q4_K_M on 7900 XTX

| Shape | Count | Speedup |
|-------|------:|--------:|
| Q4_K 4096x12288 | 72 | **1.94x** |
| Q6_K 4096x1024 | 18 | **2.10x** |
| Q4_K 4096x4096 | 72 | **1.21x** |

## llama.cpp Patch

A ~50 line patch to `mmvq.cu` adds runtime shape-specific config loading:

1. `smithy-config.h` -- config struct + JSON loader, lazy-loads on first kernel dispatch
2. `smithy_lookup()` in `calc_launch_params()` -- overrides nwarps/rows_per_block per shape
3. Falls back to existing defaults when no config exists

Branch: `smithy-shape-configs` in the llama.cpp fork.

## Architecture

```
kernel-anvil gguf-optimize model.gguf
    |
    +-- Parse GGUF (tensor shapes + quant types)
    +-- For each unique (quant, N, K):
    |     Profile -> Classify bottleneck -> Sweep configs -> Verify
    +-- Write JSON to ~/.cache/smithy/<model>.json
    +-- llama.cpp reads at first kernel dispatch
```

## Bottleneck Classifications

| Class | Condition | Optimization |
|-------|-----------|-------------|
| bandwidth_bound | BW util > 60% | Larger BLOCK_K, try SPLIT_K |
| occupancy_limited_vgpr | Occupancy < 50% | Lower BLOCK_N, fewer warps |
| occupancy_limited_lds | LDS limits waves | Reduce shared memory, num_stages=1 |
| register_spill | scratch > 0 | Smaller blocks |
| compute_bound | Low BW, high occupancy | Larger blocks, more warps |

## Supported GPUs

| Family | GPUs | Status |
|--------|------|--------|
| RDNA 2 | RX 6900 XT, RX 6800 XT/6800, RX 6700 XT/6650/6600 | Supported |
| RDNA 3 | RX 7900 XTX/XT, RX 7800 XT, RX 7700 XT | Tested |
| RDNA 3.5 | Radeon AI 370/395 (Strix Halo), Strix Point | Supported |
| RDNA 4 | RX 9070 XT, RX 9070, R9700 AI Pro (32GB) | Supported |

The profiling + sweep runs on any GPU that supports PyTorch + Triton. Hardware specs for occupancy analysis are built-in for all listed GPUs.

## Limitations

- **AMD only** (CUDA/Metal support planned)
- **MMVQ decode path only** (batch=1 GEMV, not prefill GEMM)
- **Env var required** for llama.cpp config loading (`SMITHY_CONFIG`)

## Testing

```bash
python -m pytest tests/ -v   # 193 tests
```

## Related Work

- [KernelSkill](https://arxiv.org/abs/2603.10085), [CUDA Agent](https://arxiv.org/abs/2602.24286), [KernelFoundry](https://arxiv.org/abs/2603.12440), [TritonForge](https://arxiv.org/abs/2512.09196) -- all target NVIDIA exclusively

kernel-anvil is the first tool targeting AMD/RDNA3.

## License

Apache-2.0
