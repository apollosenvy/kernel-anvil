"""C header code generation for llama.cpp kernel config tables.

Takes sweep results mapping (quant_type, N, K) -> {nwarps, rows_per_block}
and emits a C header with bucketed lookup tables that llama.cpp can use
to select per-shape kernel configs at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


# Bucket boundaries for N and K dimensions.
# A value is placed in the first bucket whose boundary it does not exceed.
# The last bucket catches everything larger.
BUCKET_BOUNDARIES = (128, 1024, 4096, 16384)
NUM_BUCKETS = len(BUCKET_BOUNDARIES) + 1  # +1 for the >max bucket

# ggml_type enum values from ggml.h -- maps quant type name to enum index.
# This determines which slot in smithy_configs[GGML_TYPE_COUNT][5][5] each type occupies.
GGML_TYPE_MAP = {
    "F32": 0, "F16": 1,
    "Q4_0": 2, "Q4_1": 3, "Q5_0": 6, "Q5_1": 7, "Q8_0": 8, "Q8_1": 9,
    "Q2_K": 10, "Q3_K": 11, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14, "Q8_K": 15,
    "IQ2_XXS": 16, "IQ2_XS": 17, "IQ3_XXS": 18, "IQ1_S": 19, "IQ4_NL": 20,
    "IQ3_S": 21, "IQ2_S": 22, "IQ4_XS": 23, "IQ1_M": 24,
    "BF16": 30,
    "MXFP4": 39, "NVFP4": 40,
}
GGML_TYPE_COUNT = 41  # must match ggml.h GGML_TYPE_COUNT

@dataclass(frozen=True)
class ShapeConfig:
    """Kernel config for a single (quant, N, K) workload."""
    nwarps: int
    rows_per_block: int


# Default config when no sweep data is available for a bucket.
DEFAULT_CONFIG = ShapeConfig(nwarps=4, rows_per_block=1)


def bucket_index(value: int) -> int:
    """Return the bucket index for a dimension value.

    Bucket 0: value <= 128
    Bucket 1: 128 < value <= 1024
    Bucket 2: 1024 < value <= 4096
    Bucket 3: 4096 < value <= 16384
    Bucket 4: value > 16384
    """
    for i, boundary in enumerate(BUCKET_BOUNDARIES):
        if value <= boundary:
            return i
    return len(BUCKET_BOUNDARIES)


def _bucket_label(idx: int) -> str:
    """Human-readable label for a bucket index."""
    if idx == 0:
        return f"<={BUCKET_BOUNDARIES[0]}"
    if idx < len(BUCKET_BOUNDARIES):
        return f"<={BUCKET_BOUNDARIES[idx]}"
    return f">{BUCKET_BOUNDARIES[-1]}"


def build_config_tables(
    configs: dict[tuple[str, int, int], dict],
    priorities: dict[tuple[str, int, int], float] | None = None,
) -> dict[str, list[list[ShapeConfig]]]:
    """Build bucketed config tables from sweep results.

    Multiple (qt, N, K) triples can land in the same (qt, n_bucket, k_bucket)
    cell. By default, the last (qt, N, K) entry that maps to a given cell
    wins (iteration order over the configs dict). When ``priorities`` is
    provided, the highest-priority entry wins per cell -- use the profiled
    speedup so the better-performing config survives the bucket collision
    (relevant for merging configs from multiple models, e.g. speculative
    decoding target + draft pairs). NaN or missing priorities are coerced
    to -inf so well-defined priorities always win over malformed ones.

    Args:
        configs: Mapping of (quant_type, N, K) -> {"nwarps": int, "rows_per_block": int}.
            The quant_type should match one of QUANT_TYPES (e.g. "Q4_K").
        priorities: Optional parallel mapping of (quant_type, N, K) -> float
            (typically profiled speedup). When two entries collide on a bucket
            cell, the one with the higher priority wins. Missing keys are
            treated as priority -inf.

    Returns:
        Dict mapping quant_type -> 2D list [n_bucket][k_bucket] of ShapeConfig.
        Only quant types present in the input are included.
    """
    tables: dict[str, list[list[ShapeConfig]]] = {}

    present_types = sorted({qt for qt, _, _ in configs})

    for qt in present_types:
        table = [[DEFAULT_CONFIG] * NUM_BUCKETS for _ in range(NUM_BUCKETS)]
        cell_priority: dict[tuple[int, int], float] = {}

        for (q, n, k), cfg in configs.items():
            if q != qt:
                continue
            ni = bucket_index(n)
            ki = bucket_index(k)
            shape_cfg = ShapeConfig(
                nwarps=cfg["nwarps"],
                rows_per_block=cfg["rows_per_block"],
            )
            if priorities is not None:
                p = priorities.get((q, n, k))
                # NaN must NOT win or lose silently: NaN compares False both
                # ways, which would let it always overwrite a real priority.
                # Treat NaN/None as -inf so well-defined priorities always
                # win over malformed ones.
                if p is None:
                    p_val = float("-inf")
                else:
                    pf = float(p)
                    p_val = float("-inf") if pf != pf else pf  # pf != pf <=> NaN
                if (ni, ki) in cell_priority and p_val <= cell_priority[(ni, ki)]:
                    continue
                cell_priority[(ni, ki)] = p_val
            table[ni][ki] = shape_cfg

        tables[qt] = table

    return tables


def generate_runtime_config(
    configs: dict[tuple[str, int, int], dict],
    gpu_name: str = "gfx1100",
    model_name: str = "unknown",
    priorities: dict[tuple[str, int, int], float] | None = None,
) -> str:
    """Generate a JSON config file for llama.cpp runtime loading.

    This is the runtime equivalent of generate_config_header(). Instead of
    a C header that requires recompilation, it produces a JSON file that
    llama.cpp's smithy-config.h loads at startup.

    The JSON format matches what smithy_load_configs() in smithy-config.h parses:
    {
      "gpu": "gfx1100",
      "model": "Qwen3-8B-Q4_K_M",
      "configs": {
        "12": {  // ggml_type enum value for Q4_K
          "2,1": {"nwarps": 8, "rows_per_block": 2},
          ...
        }
      }
    }

    When ``priorities`` is provided (typically per-shape profiled speedup),
    bucket-cell collisions resolve to the higher-priority entry. This is how
    speculative-decoding configs from a target+draft pair are merged.
    """
    import json

    tables = build_config_tables(configs, priorities=priorities)

    out = {"gpu": gpu_name, "model": model_name, "configs": {}}

    for qt, table in tables.items():
        type_idx = GGML_TYPE_MAP.get(qt)
        if type_idx is None:
            continue
        type_configs = {}
        for ni in range(NUM_BUCKETS):
            for ki in range(NUM_BUCKETS):
                cfg = table[ni][ki]
                if cfg != DEFAULT_CONFIG:
                    type_configs[f"{ni},{ki}"] = {
                        "nwarps": cfg.nwarps,
                        "rows_per_block": cfg.rows_per_block,
                    }
        if type_configs:
            out["configs"][str(type_idx)] = type_configs

    return json.dumps(out, indent=2)


def merge_runtime_configs(
    payloads: list[dict],
    gpu_name: str | None = None,
    model_name: str | None = None,
) -> dict:
    """Merge already-serialized runtime config payloads (loaded from JSON).

    Used to combine configs from multiple ``gguf-optimize`` runs (e.g. a
    speculative-decoding target + draft pair) into a single config that
    llama.cpp loads via SMITHY_CONFIG.

    Resolution policy: first-seen wins. Earlier payloads in ``payloads`` take
    priority over later ones for any overlapping (type_idx, n_bucket, k_bucket)
    cell. Callers control priority via argument order -- pass the most
    important model (typically the target) first.

    Malformed payloads (non-dict at any level: payload, configs map, per-type
    map, individual cell config) are silently skipped rather than raising,
    so partial / mistyped JSON degrades to omitted entries instead of
    crashing the whole merge.

    Args:
        payloads: List of parsed JSON payloads, each shaped like
            ``{"gpu": ..., "model": ..., "configs": {type_idx: {"n,k": {...}}}}``.
        gpu_name: Override for the merged ``gpu`` field. If None, uses the
            first payload's value (or "unknown").
        model_name: Override for the merged ``model`` field. If None, joins
            the input model names with ``+``.

    Returns:
        A merged payload dict in the same shape, ready for ``json.dumps``.
    """
    if not payloads:
        return {"gpu": gpu_name or "unknown", "model": model_name or "merged", "configs": {}}

    # Defensively skip malformed payloads / configs / cells rather than
    # raising an AttributeError into the caller. The CLI loads these from
    # user-supplied JSON, so partial / mistyped data must degrade to "skip
    # this entry" instead of crashing the whole merge.
    merged_configs: dict[str, dict[str, dict]] = {}
    valid_payloads: list[dict] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        valid_payloads.append(payload)
        cfgs = payload.get("configs") or {}
        if not isinstance(cfgs, dict):
            continue
        for type_idx, cells in cfgs.items():
            if not isinstance(cells, dict):
                continue
            type_bucket = merged_configs.setdefault(str(type_idx), {})
            for cell_key, cell_cfg in cells.items():
                if cell_key in type_bucket:
                    continue  # first-seen wins
                if not isinstance(cell_cfg, dict):
                    continue
                type_bucket[str(cell_key)] = dict(cell_cfg)

    if gpu_name is not None:
        out_gpu = gpu_name
    else:
        gpu_field = valid_payloads[0].get("gpu") if valid_payloads else None
        out_gpu = gpu_field if isinstance(gpu_field, str) else "unknown"

    if model_name is not None:
        out_model = model_name
    else:
        names = []
        for p in valid_payloads:
            m = p.get("model")
            names.append(m if isinstance(m, str) else "unknown")
        out_model = "+".join(names) if names else "merged"

    return {"gpu": out_gpu, "model": out_model, "configs": merged_configs}


def generate_config_header(
    configs: dict[tuple[str, int, int], dict],
    gpu_name: str = "gfx1100 (7900 XTX)",
    model_name: str = "unknown",
) -> str:
    """Generate a C header compatible with llama.cpp's mmvq.cu smithy patch.

    Emits smithy_configs[GGML_TYPE_COUNT][SMITHY_NUM_BUCKETS][SMITHY_NUM_BUCKETS]
    indexed by ggml_type enum values so mmvq.cu can look up configs directly.

    Args:
        configs: Mapping of (quant_type, N, K) -> {"nwarps": int, "rows_per_block": int}.
        gpu_name: GPU identifier for the header comment.
        model_name: Model name for the header comment.

    Returns:
        Complete C header file content as a string.
    """
    tables = build_config_tables(configs)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append(f"// smithy-config.h - Generated by kernel-anvil for {gpu_name}")
    lines.append(f"// Model: {model_name}")
    lines.append(f"// Generated: {timestamp}")
    lines.append("//")
    lines.append("// Bucket boundaries: " + ", ".join(str(b) for b in BUCKET_BOUNDARIES))
    lines.append(f"// Buckets per axis: {NUM_BUCKETS}")
    lines.append("//")
    lines.append("// Drop this file into ggml/src/ggml-cuda/ to replace the default smithy-config.h")
    lines.append("// Rebuild llama.cpp to apply shape-specific kernel configs.")
    lines.append("#pragma once")
    lines.append("")
    lines.append("struct smithy_shape_config {")
    lines.append("    int nwarps;         // 0 = use default (no override)")
    lines.append("    int rows_per_block; // 0 = use default (no override)")
    lines.append("};")
    lines.append("")

    # Bucket enum matching kernel-researcher's format
    lines.append("enum smithy_bucket : int {")
    for i, boundary in enumerate(BUCKET_BOUNDARIES):
        lines.append(f"    SMITHY_BUCKET_LE_{boundary:<5d} = {i},")
    lines.append(f"    SMITHY_BUCKET_GT_{BUCKET_BOUNDARIES[-1]:<5d} = {len(BUCKET_BOUNDARIES)},")
    lines.append(f"    SMITHY_NUM_BUCKETS     = {NUM_BUCKETS},")
    lines.append("};")
    lines.append("")

    # Bucket lookup function
    lines.append("static inline smithy_bucket smithy_get_bucket(int value) {")
    for i, boundary in enumerate(BUCKET_BOUNDARIES):
        lines.append(f"    if (value <= {boundary}) return static_cast<smithy_bucket>({i});")
    lines.append(f"    return static_cast<smithy_bucket>({len(BUCKET_BOUNDARIES)});")
    lines.append("}")
    lines.append("")

    # Build full GGML_TYPE_COUNT table -- zero for types without sweep data
    lines.append(f"// Indexed by [ggml_type][N_bucket][K_bucket]")
    lines.append(f"// Non-zero entries override llama.cpp defaults. Zero = no override.")
    lines.append(f"#ifndef SMITHY_CONFIG_TABLE")
    lines.append(f"#define SMITHY_CONFIG_TABLE")
    lines.append(f"static const smithy_shape_config smithy_configs[{GGML_TYPE_COUNT}][SMITHY_NUM_BUCKETS][SMITHY_NUM_BUCKETS] = {{")

    for type_idx in range(GGML_TYPE_COUNT):
        # Find which quant type this enum value corresponds to
        qt_name = None
        for name, idx in GGML_TYPE_MAP.items():
            if idx == type_idx:
                qt_name = name
                break

        if qt_name and qt_name in tables:
            table = tables[qt_name]
            lines.append(f"    // [{type_idx}] = GGML_TYPE_{qt_name}")
            lines.append("    {")
            for ni in range(NUM_BUCKETS):
                row_entries = []
                for ki in range(NUM_BUCKETS):
                    cfg = table[ni][ki]
                    # Use 0 for default (no override), actual values for profiled entries
                    nw = cfg.nwarps if cfg != DEFAULT_CONFIG else 0
                    rpb = cfg.rows_per_block if cfg != DEFAULT_CONFIG else 0
                    row_entries.append(f"{{{nw:2d},{rpb:2d}}}")
                lines.append(f"        {{{', '.join(row_entries)}}},")
            lines.append("    },")
        else:
            lines.append(f"    {{{{}}}},  // [{type_idx}] unused")

    lines.append("};")
    lines.append("#endif // SMITHY_CONFIG_TABLE")
    lines.append("")

    # Summary comment
    lines.append("// Profiled types:")
    for qt in sorted(tables):
        ggml_idx = GGML_TYPE_MAP.get(qt, "?")
        lines.append(f"//   GGML_TYPE_{qt} = {ggml_idx}")
    lines.append("")

    return "\n".join(lines)
