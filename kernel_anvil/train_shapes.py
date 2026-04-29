"""Extract Triton-tunable GEMM shapes from a Hugging Face model config.

Inference-side autotune (`gguf-optimize`) parses GGUF tensors directly to
discover unique (N, K) projection shapes. Training has no GGUF; we walk the
HF model config and synthesize the shapes the forward + grad-input kernels
will see for a given (batch, seq) pair.

Supported architectures:

- **Dense LLaMA / Qwen3** (`LlamaConfig`, `Qwen3Config`): four forward
  shapes per layer (q_proj, k_proj, v_proj/o_proj, gate_up_proj, down_proj
  -- collapsed to four uniques).
- **gpt-oss MoE** (`GptOssConfig`): attention identical to dense; experts
  contribute irregular-M shapes flattened to a single (M, N=2*intermediate,
  K=hidden) gate_up and (M, N=hidden, K=intermediate) down_proj. The bucket
  scheme handles the irregular-M side.

The output is the deduplicated set of (op, M, N, K) tuples, where `op`
spans the requested ops list (default: forward + grad-input for both
quants).
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

DEFAULT_OPS: tuple[str, ...] = (
    "mxfp4_fwd",
    "mxfp4_grad_input",
    "int4_fwd",
    "int4_grad_input",
)


def _load_hf_config(model_id_or_config) -> object:
    """Load an HF AutoConfig, with graceful fallback for offline / cache.

    Returns the config object directly if already an instance. Otherwise
    invokes `transformers.AutoConfig.from_pretrained(..., trust_remote_code
    =True)`. If the fetch fails (no network, private repo, missing local
    cache), retries with `local_files_only=True` to lean on the user's
    huggingface cache. If even that misses, raises a clear RuntimeError so
    the CLI can surface a useful message.
    """
    if model_id_or_config is None:
        raise ValueError("model_id_or_config must not be None")
    # Allow passing a pre-built config object (test fixture, custom config).
    if not isinstance(model_id_or_config, (str, os.PathLike)):
        return model_id_or_config

    try:
        from transformers import AutoConfig  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed; install with `pip install transformers`"
            " or pass a pre-built config object."
        ) from exc

    model_id = str(model_id_or_config)
    last_err: Exception | None = None
    for kwargs in ({"trust_remote_code": True}, {"trust_remote_code": True, "local_files_only": True}):
        try:
            return AutoConfig.from_pretrained(model_id, **kwargs)
        except Exception as exc:  # network errors, gated repos, missing cache
            last_err = exc
    raise RuntimeError(
        f"Could not load HF config for {model_id!r} (network and local cache both failed): {last_err}"
    )


def _arch_name(cfg) -> str:
    """Best-effort architecture name string for routing.

    Real `transformers` config objects expose their class as
    `type(cfg).__name__` (e.g. ``LlamaConfig``). For test fixtures we also
    accept a stashed ``__class__`` attribute pointing at a class object
    whose name carries the arch hint -- this lets us exercise dispatch
    without constructing real transformers configs.
    """
    real = type(cfg).__name__
    if real and real not in ("SimpleNamespace", "object", "dict"):
        return real
    stash = getattr(cfg, "__class__", None)
    stash_name = getattr(stash, "__name__", "") if stash is not None else ""
    if stash_name and stash_name not in ("SimpleNamespace", "object", "dict", "type"):
        return stash_name
    archs = getattr(cfg, "architectures", None) or []
    if archs:
        return str(archs[0])
    # model_type is a transformers.PretrainedConfig field that survives
    # most subclassing patterns.
    return str(getattr(cfg, "model_type", "") or "")


def _is_llama_like(cfg) -> bool:
    name = _arch_name(cfg).lower()
    return "llama" in name or "qwen" in name or "mistral" in name or "phi" in name


def _is_gpt_oss(cfg) -> bool:
    name = _arch_name(cfg).lower()
    return "gptoss" in name or "gpt_oss" in name or "gpt-oss" in name


def _dense_shapes(cfg, M: int) -> list[tuple[int, int, int]]:
    """Forward GEMM shapes for a dense LLaMA/Qwen-style decoder layer.

    Per-layer ops:
        q_proj:       N=hidden,             K=hidden
        k_proj/v_proj:N=num_kv_heads*head_dim, K=hidden  (GQA may collapse to one)
        o_proj:       N=hidden,             K=hidden
        gate/up_proj: N=intermediate,       K=hidden
        down_proj:    N=hidden,             K=intermediate

    The unique shapes (deduped) for a single forward pass:
        (M, hidden, hidden)               -- q_proj, o_proj
        (M, kv_dim, hidden)               -- k_proj, v_proj when kv_dim != hidden
        (M, intermediate, hidden)         -- gate, up
        (M, hidden, intermediate)         -- down
    """
    hidden = int(getattr(cfg, "hidden_size"))
    intermediate = int(getattr(cfg, "intermediate_size", hidden * 4))
    num_heads = int(getattr(cfg, "num_attention_heads", 1))
    num_kv_heads = int(
        getattr(cfg, "num_key_value_heads", num_heads) or num_heads
    )
    head_dim = int(getattr(cfg, "head_dim", hidden // max(num_heads, 1)))
    kv_dim = num_kv_heads * head_dim

    shapes: list[tuple[int, int, int]] = [
        (M, hidden, hidden),
        (M, intermediate, hidden),
        (M, hidden, intermediate),
    ]
    if kv_dim and kv_dim != hidden:
        shapes.append((M, kv_dim, hidden))
    return shapes


def _gpt_oss_shapes(cfg, M: int) -> list[tuple[int, int, int]]:
    """Forward GEMM shapes for a gpt-oss MoE decoder layer.

    Attention shapes are dense-equivalent. Expert shapes are flattened across
    experts -- we treat the tokens-per-expert as irregular-M; the 3D bucket
    scheme handles that. The MoE-specific shapes are:

        gate_up_proj: N = 2 * intermediate, K = hidden
        down_proj:    N = hidden,           K = intermediate

    (`2 *` because gate_up packs gate || up into a single weight tile.)
    """
    hidden = int(getattr(cfg, "hidden_size"))
    intermediate = int(getattr(cfg, "intermediate_size", hidden * 4))
    num_experts = int(getattr(cfg, "num_local_experts", getattr(cfg, "num_experts", 1)) or 1)
    top_k = int(getattr(cfg, "num_experts_per_tok", 1) or 1)

    # Dense attention shapes (q,k,v,o) reuse the dense logic.
    attn_shapes = _dense_shapes(cfg, M)

    # Expert M: each token is routed to top_k experts; per expert the M is
    # roughly M * top_k / num_experts. We round up to nearest power of two
    # to keep the bucket count small. The 3D bucket scheme buckets the
    # irregular-M cases together, so the actual runtime M can vary.
    if num_experts > 0 and top_k > 0:
        per_expert_m = max(1, (M * top_k + num_experts - 1) // num_experts)
    else:
        per_expert_m = M

    expert_shapes = [
        (per_expert_m, 2 * intermediate, hidden),  # gate_up
        (per_expert_m, hidden, intermediate),      # down
    ]
    return attn_shapes + expert_shapes


def _shapes_for_config(cfg, M: int) -> list[tuple[int, int, int]]:
    """Dispatch shape extraction based on architecture string."""
    if _is_gpt_oss(cfg):
        return _gpt_oss_shapes(cfg, M)
    if _is_llama_like(cfg):
        return _dense_shapes(cfg, M)
    # Generic fallback: assume LLaMA-like attribute names. transformers is
    # consistent enough that hidden_size + intermediate_size are present on
    # almost every decoder config.
    if hasattr(cfg, "hidden_size") and hasattr(cfg, "intermediate_size"):
        return _dense_shapes(cfg, M)
    raise ValueError(
        f"Unsupported HF config: {_arch_name(cfg)} -- add a shape extractor"
        " or pass a known architecture (LlamaConfig, Qwen3Config, GptOssConfig)."
    )


def extract_shapes(
    model_id_or_config,
    *,
    batch: int,
    seq: int,
    ops: Sequence[str] | None = None,
) -> list[tuple[str, int, int, int]]:
    """Return the deduped list of (op, M, N, K) tuples for the given model.

    Args:
        model_id_or_config: HF model id (string), local path, or pre-built
            transformers config object.
        batch: Per-device batch size used at training time.
        seq: Sequence length used at training time.
        ops: Iterable of op names. Defaults to all four (mxfp4 fwd/grad-in,
            int4 fwd/grad-in). The same (M, N, K) shape is emitted once per
            requested op.

    Returns:
        Deduplicated list of (op_name, M, N, K) tuples in stable order
        (sorted on the tuple).
    """
    if batch <= 0 or seq <= 0:
        raise ValueError(f"batch and seq must be positive (got batch={batch}, seq={seq})")
    if ops is None:
        op_list = list(DEFAULT_OPS)
    else:
        op_list = list(ops)
        if not op_list:
            raise ValueError("ops must be a non-empty iterable")

    cfg = _load_hf_config(model_id_or_config)
    M = batch * seq
    raw_shapes = _shapes_for_config(cfg, M)

    seen: set[tuple[str, int, int, int]] = set()
    out: list[tuple[str, int, int, int]] = []
    for op in op_list:
        for m, n, k in raw_shapes:
            if m <= 0 or n <= 0 or k <= 0:
                continue
            key = (op, int(m), int(n), int(k))
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    out.sort()
    return out


def model_basename(model_id: str) -> str:
    """Best-effort model basename for cache-path construction.

    `Qwen/Qwen3-8B` -> `Qwen3-8B`. Local paths use the directory stem.
    """
    if not model_id:
        return "model"
    s = str(model_id).rstrip("/")
    if "/" in s:
        return s.split("/")[-1]
    return s
