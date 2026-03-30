"""Model-level auto-optimization: replace nn.Linear with Triton GEMV.

Usage:
    import kernel_anvil
    model = AutoModelForCausalLM.from_pretrained(...)
    kernel_anvil.optimize(model)
    # model now uses optimized Triton GEMV for single-token decode
"""
import hashlib
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernel_anvil.analyze import classify
from kernel_anvil.profile import profile_kernel
from kernel_anvil.rdna3 import GFX1100, detect_gpu
from kernel_anvil.sweep import generate_configs
from kernel_anvil.verify import verify_and_bench


# ---------------------------------------------------------------------------
# SmithyLinear -- drop-in nn.Linear replacement
# ---------------------------------------------------------------------------


class SmithyLinear(nn.Module):
    """Drop-in replacement for nn.Linear using optimized Triton GEMV.

    For single-token decode (total tokens == 1), launches a Triton GEMV kernel
    tuned for the specific (out_features, in_features) shape on RDNA3.
    For batched input (prefill, total tokens > 1), falls back to
    torch.nn.functional.linear (cuBLAS/hipBLAS).
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, config: dict):
        super().__init__()
        # Store as parameters so .to() / .half() / .cuda() propagate
        self.weight = nn.Parameter(weight, requires_grad=weight.requires_grad)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=bias.requires_grad)
        else:
            self.bias = None
        self.config = config
        self.out_features, self.in_features = weight.shape

    def _total_tokens(self, x: torch.Tensor) -> int:
        """Compute the total number of token positions in input."""
        if x.dim() == 3:
            return x.shape[0] * x.shape[1]  # batch * seq_len
        if x.dim() == 2:
            return x.shape[0]
        # 1D: single vector
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._total_tokens(x) == 1 and x.is_cuda:
            return self._triton_gemv(x)
        return F.linear(x, self.weight, self.bias)

    def _triton_gemv(self, x: torch.Tensor) -> torch.Tensor:
        """Run the Triton GEMV kernel on a single input vector."""
        from kernel_anvil.kernels import triton_gemv

        orig_shape = x.shape
        # Flatten to 1D vector of length K
        x_flat = x.reshape(-1).contiguous()
        out = triton_gemv(x_flat, self.weight, self.bias, self.config)

        # Restore leading dims: [batch, seq_len, out_features] or [1, out_features]
        if len(orig_shape) == 3:
            return out.reshape(orig_shape[0], orig_shape[1], self.out_features)
        if len(orig_shape) == 2:
            return out.reshape(1, self.out_features)
        return out

    def extra_repr(self) -> str:
        bias_str = "bias=True" if self.bias is not None else "bias=False"
        cfg_str = " ".join(f"{k}={v}" for k, v in sorted(self.config.items()))
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"{bias_str}, config=({cfg_str})"
        )


# ---------------------------------------------------------------------------
# Model walker helpers
# ---------------------------------------------------------------------------


def _find_linears(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Find all nn.Linear modules in the model tree.

    Returns list of (dotted_name, module) pairs.
    Only returns actual nn.Linear instances, not SmithyLinear.
    """
    result = []
    for name, module in model.named_modules():
        if type(module) is nn.Linear:
            result.append((name, module))
    return result


def _replace_linear(model: nn.Module, name: str, smithy_linear: SmithyLinear):
    """Replace a named module in the model tree with a SmithyLinear."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], smithy_linear)


# ---------------------------------------------------------------------------
# Config caching
# ---------------------------------------------------------------------------


def _cache_path(model: nn.Module, cache_dir: str | None) -> str:
    """Generate cache file path from model config or structure hash."""
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "model_type"):
        hidden = getattr(config, "hidden_size", 0)
        layers = getattr(config, "num_hidden_layers", 0)
        key = f"{config.model_type}_{hidden}_{layers}"
    else:
        key = hashlib.md5(str(model).encode()).hexdigest()[:16]
    cache_dir = cache_dir or os.path.expanduser("~/.cache/smithy")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{key}.json")


def _save_configs(path: str, configs: dict):
    """Save shape->config mapping to JSON."""
    with open(path, "w") as f:
        json.dump(configs, f, indent=2)


def _load_configs(path: str) -> dict | None:
    """Load cached shape->config mapping, or None if absent."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Per-shape tuning
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "BLOCK_N": 64,
    "BLOCK_K": 128,
    "num_warps": 4,
    "num_stages": 1,
}


def _tune_shape(
    N: int,
    K: int,
    device: torch.device,
    gpu_spec=None,
    warmup: int = 3,
    runs: int = 10,
    max_configs: int = 12,
    verbose: bool = True,
) -> dict:
    """Profile and sweep to find the optimal GEMV config for shape (N, K).

    Returns the winning config dict.
    """
    from kernel_anvil.kernels import triton_gemv

    # Create test tensors
    x = torch.randn(K, device=device, dtype=torch.float16)
    w = torch.randn(N, K, device=device, dtype=torch.float16)
    bias = None  # tune without bias (config is shape-dependent, not bias-dependent)

    # Reference output for correctness checks
    ref = F.linear(x.unsqueeze(0), w).squeeze(0)

    # Data bytes for bandwidth estimation: read W[N,K] + x[K], write out[N]
    data_bytes = (N * K + K + N) * 2  # FP16 = 2 bytes

    baseline_config = dict(_DEFAULT_CONFIG)

    def kernel_fn(config):
        return triton_gemv(x, w, bias, config)

    # Profile baseline
    if gpu_spec is None:
        gpu_spec = GFX1100

    metrics = profile_kernel(
        kernel_fn=kernel_fn,
        config=baseline_config,
        data_bytes=data_bytes,
        gpu_spec=gpu_spec,
        warmup=warmup,
        runs=runs,
    )

    report = classify(metrics, gpu_spec)

    # Generate candidate configs
    candidates = generate_configs(
        report,
        baseline_config=baseline_config,
        max_configs=max_configs,
    )

    if not candidates:
        return baseline_config

    # Benchmark baseline for speedup reference
    baseline_result = verify_and_bench(
        kernel_fn=kernel_fn,
        reference_output=ref,
        config=baseline_config,
        warmup=warmup,
        runs=runs,
        data_bytes=data_bytes,
    )

    # Sweep candidates
    best_config = baseline_config
    best_latency = baseline_result.latency_us

    for cfg in candidates:
        # Filter out SPLIT_K since our GEMV kernel doesn't support it
        cfg_clean = {k: v for k, v in cfg.items() if k != "SPLIT_K"}
        try:
            result = verify_and_bench(
                kernel_fn=kernel_fn,
                reference_output=ref,
                config=cfg_clean,
                warmup=warmup,
                runs=runs,
                data_bytes=data_bytes,
                baseline_latency_us=baseline_result.latency_us,
                atol=1e-2,
                rtol=1e-2,
            )
            if result.correct and result.latency_us < best_latency:
                best_latency = result.latency_us
                best_config = cfg_clean
        except Exception:
            # Config may trigger invalid Triton parameters, skip
            continue

    return best_config


# ---------------------------------------------------------------------------
# Main API: optimize()
# ---------------------------------------------------------------------------


def optimize(
    model: nn.Module,
    warmup_tokens: torch.Tensor | None = None,
    cache_dir: str | None = None,
    verbose: bool = True,
    _force_tune: bool = False,
) -> nn.Module:
    """Auto-optimize a model's nn.Linear layers with Triton GEMV.

    Walks the module tree, finds all nn.Linear layers, groups them by shape,
    profiles and tunes a Triton GEMV config for each unique shape, then
    replaces every nn.Linear with a SmithyLinear that uses the winning config.

    Args:
        model: Any nn.Module (typically a HuggingFace CausalLM).
        warmup_tokens: Optional input_ids for a warmup pass (unused currently,
            reserved for future JIT warmup).
        cache_dir: Where to save/load config cache. Default: ~/.cache/smithy/
        verbose: Print progress info.
        _force_tune: If True, skip cache and always re-tune (for testing).

    Returns:
        The model (modified in-place) with Linear layers replaced.
    """
    linears = _find_linears(model)
    if not linears:
        if verbose:
            print("[kernel-anvil] No nn.Linear layers found.")
        return model

    # Try loading cached configs
    cache_file = _cache_path(model, cache_dir)
    cached = None
    if not _force_tune:
        cached = _load_configs(cache_file)

    if cached is not None and verbose:
        print(f"[kernel-anvil] Loaded cached configs from {cache_file}")

    # Group by (out_features, in_features) shape
    shapes: dict[tuple[int, int], list[tuple[str, nn.Linear]]] = {}
    for name, mod in linears:
        key = (mod.out_features, mod.in_features)
        shapes.setdefault(key, []).append((name, mod))

    if verbose:
        print(
            f"[kernel-anvil] Found {len(linears)} nn.Linear layers "
            f"({len(shapes)} unique shapes)"
        )

    # Determine device from first linear layer's weight
    device = linears[0][1].weight.device
    has_gpu = device.type == "cuda"

    # Detect GPU spec for tuning
    gpu_spec = None
    if has_gpu:
        gpu_spec = detect_gpu()
    if gpu_spec is None:
        gpu_spec = GFX1100

    # Resolve config for each unique shape
    shape_configs: dict[str, dict] = {}
    for (N, K), members in shapes.items():
        shape_key = f"({N}, {K})"

        if cached and shape_key in cached:
            config = cached[shape_key]
        elif has_gpu and not cached:
            if verbose:
                print(f"[kernel-anvil] Tuning shape {shape_key}...", end=" ", flush=True)
            t0 = time.monotonic()
            config = _tune_shape(N, K, device, gpu_spec=gpu_spec, verbose=verbose)
            dt = time.monotonic() - t0
            if verbose:
                cfg_str = " ".join(f"{k}={v}" for k, v in sorted(config.items()))
                print(f"done ({dt:.1f}s) -> {cfg_str}")
        else:
            # No GPU or no cache -- use defaults
            config = dict(_DEFAULT_CONFIG)

        shape_configs[shape_key] = config

    # Replace all Linear layers
    replaced = 0
    for (N, K), members in shapes.items():
        shape_key = f"({N}, {K})"
        config = shape_configs[shape_key]
        for name, mod in members:
            smithy_mod = SmithyLinear(mod.weight.data, mod.bias.data if mod.bias is not None else None, config)
            _replace_linear(model, name, smithy_mod)
            replaced += 1

    if verbose:
        print(f"[kernel-anvil] Replaced {replaced} nn.Linear layers with SmithyLinear")

    # Save configs to cache
    if not cached or _force_tune:
        _save_configs(cache_file, shape_configs)
        if verbose:
            print(f"[kernel-anvil] Saved configs to {cache_file}")

    return model
