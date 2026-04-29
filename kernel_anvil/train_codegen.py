"""Code generation for the anvil-train JSON v1 schema.

The training-side counterpart of `codegen.py`. Where the inference codegen
emits llama.cpp `smithy_configs` C-style tables keyed on (ggml_type, N, K),
this module emits a JSON contract consumed by mud-puppy's anvil_loader and
keyed on (op_name, M, N, K) -- a 3D bucket scheme.

Schema (anvil-train/v1):

```json
{
  "schema": "anvil-train/v1",
  "gpu": "gfx1100",
  "rocm_version": "7.1",
  "torch_version": "2.10.0+rocm7.1",
  "triton_version": "3.6.0",
  "kernel_hash": "sha256:<hex>",
  "model": "Qwen3-8B",
  "batch": 1,
  "seq": 4096,
  "ops": {
    "<op_name>": {
      "<m_bucket>,<n_bucket>,<k_bucket>": {
        "BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32,
        "GROUP_M": 8, "num_warps": 8, "num_stages": 4,
        "speedup_vs_baseline": 1.33,
        "profiled_us": 18.4
      }
    }
  }
}
```

All bucket boundaries are (1024, 4096, 8192, 16384) -- five buckets per axis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

# Schema identifier embedded in every emitted payload. Bump when the on-disk
# layout changes incompatibly.
SCHEMA_VERSION = "anvil-train/v1"

# Bucket boundaries shared across the M, N, and K axes. A value falls into the
# first bucket whose boundary it does not exceed; everything larger lands in
# the final >max bucket.
BUCKET_BOUNDARIES = (1024, 4096, 8192, 16384)
NUM_BUCKETS = len(BUCKET_BOUNDARIES) + 1  # +1 for the >max bucket

# Required keys for every per-cell payload. speedup_vs_baseline / profiled_us
# are optional metadata.
REQUIRED_PAYLOAD_KEYS = (
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "GROUP_M",
    "num_warps",
    "num_stages",
)

OPTIONAL_PAYLOAD_KEYS = (
    "speedup_vs_baseline",
    "profiled_us",
)

# Well-known op names. The schema does not enforce membership but consumers
# should use these to avoid drift.
KNOWN_OPS = (
    "mxfp4_fwd",
    "mxfp4_grad_input",
    "int4_fwd",
    "int4_grad_input",
)


@dataclass(frozen=True)
class TrainShapeConfig:
    """Per-cell payload for the training JSON.

    Mirrors `ShapeConfig` in `codegen.py` but for training kernels. The
    optional fields capture sweep telemetry; consumers may safely ignore them.
    """

    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    GROUP_M: int
    num_warps: int
    num_stages: int
    speedup_vs_baseline: float | None = None
    profiled_us: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "BLOCK_M": int(self.BLOCK_M),
            "BLOCK_N": int(self.BLOCK_N),
            "BLOCK_K": int(self.BLOCK_K),
            "GROUP_M": int(self.GROUP_M),
            "num_warps": int(self.num_warps),
            "num_stages": int(self.num_stages),
        }
        if self.speedup_vs_baseline is not None:
            out["speedup_vs_baseline"] = float(self.speedup_vs_baseline)
        if self.profiled_us is not None:
            out["profiled_us"] = float(self.profiled_us)
        return out


def bucket_index_3d(m: int, n: int, k: int) -> tuple[int, int, int]:
    """Map an (M, N, K) shape to a (mb, nb, kb) bucket triple.

    Each axis uses the same bucket boundaries, so the function is just three
    independent 1D bucket lookups. Bucket 0 covers values <= 1024, bucket 4
    catches anything > 16384.
    """
    return (_bucket_index_1d(m), _bucket_index_1d(n), _bucket_index_1d(k))


def _bucket_index_1d(value: int) -> int:
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


def _coerce_payload(raw: dict[str, Any]) -> TrainShapeConfig | None:
    """Validate a per-cell config dict and lift it to TrainShapeConfig.

    Returns None for malformed entries (missing required keys, wrong types).
    First-seen-wins is enforced by the caller; this function only validates.
    """
    if not isinstance(raw, dict):
        return None
    try:
        for key in REQUIRED_PAYLOAD_KEYS:
            if key not in raw:
                return None
            # All required fields must be ints (or int-coercible)
            int(raw[key])
        speedup = raw.get("speedup_vs_baseline")
        profiled = raw.get("profiled_us")
        return TrainShapeConfig(
            BLOCK_M=int(raw["BLOCK_M"]),
            BLOCK_N=int(raw["BLOCK_N"]),
            BLOCK_K=int(raw["BLOCK_K"]),
            GROUP_M=int(raw["GROUP_M"]),
            num_warps=int(raw["num_warps"]),
            num_stages=int(raw["num_stages"]),
            speedup_vs_baseline=float(speedup) if speedup is not None else None,
            profiled_us=float(profiled) if profiled is not None else None,
        )
    except (TypeError, ValueError):
        return None


def build_op_table(
    configs: dict[tuple[str, int, int, int], dict[str, Any]],
    priorities: dict[tuple[str, int, int, int], int] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Bucket sweep results into the on-disk op table layout.

    Args:
        configs: Mapping of (op_name, M, N, K) -> per-cell payload dict.
            Required keys: BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps,
            num_stages. Optional: speedup_vs_baseline, profiled_us.
        priorities: Optional mapping of the same key -> integer priority.
            When two shapes collide on the same (op, mb, nb, kb) cell, the
            higher-priority entry wins. With no priorities given, the
            first-seen entry wins (insertion order).

    Returns:
        Nested dict shape:
            { op_name -> { "mb,nb,kb" -> payload_dict } }

        Cells with no sweep data are simply absent (consumer falls back).
    """
    priorities = priorities or {}
    op_table: dict[str, dict[str, dict[str, Any]]] = {}
    cell_priority: dict[tuple[str, str], int] = {}

    for key, raw in configs.items():
        if not (isinstance(key, tuple) and len(key) == 4):
            continue
        op_name, m, n, k = key
        if not isinstance(op_name, str) or not op_name:
            continue
        try:
            m_i = int(m)
            n_i = int(n)
            k_i = int(k)
        except (TypeError, ValueError):
            continue
        if m_i <= 0 or n_i <= 0 or k_i <= 0:
            continue

        cfg = _coerce_payload(raw)
        if cfg is None:
            continue

        mb, nb, kb = bucket_index_3d(m_i, n_i, k_i)
        cell_key = f"{mb},{nb},{kb}"

        prio = int(priorities.get(key, 0))
        existing_prio = cell_priority.get((op_name, cell_key))

        # First-seen-wins when no priority differentiates the two entries.
        if existing_prio is None or prio > existing_prio:
            op_table.setdefault(op_name, {})[cell_key] = cfg.to_dict()
            cell_priority[(op_name, cell_key)] = prio

    return op_table


def generate_train_runtime_config(
    configs: dict[tuple[str, int, int, int], dict[str, Any]],
    *,
    gpu: str,
    model: str,
    batch: int,
    seq: int,
    rocm_version: str = "",
    torch_version: str = "",
    triton_version: str = "",
    kernel_hash: str = "",
    priorities: dict[tuple[str, int, int, int], int] | None = None,
) -> str:
    """Render the full anvil-train JSON v1 payload.

    Args:
        configs: Same shape as `build_op_table`.
        gpu: Architecture string (e.g. "gfx1100").
        model: HF model id or basename (e.g. "Qwen3-8B").
        batch: Effective M batch dimension used during the sweep.
        seq: Sequence length used during the sweep.
        rocm_version / torch_version / triton_version / kernel_hash:
            Cache-invalidation keys. Mismatch on load means re-tune.
        priorities: Optional priority map (see `build_op_table`).

    Returns:
        Pretty-printed JSON string ready to write to disk.
    """
    op_table = build_op_table(configs, priorities=priorities)

    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "gpu": gpu,
        "rocm_version": rocm_version,
        "torch_version": torch_version,
        "triton_version": triton_version,
        "kernel_hash": kernel_hash,
        "model": model,
        "batch": int(batch),
        "seq": int(seq),
        "ops": op_table,
    }
    return json.dumps(payload, indent=2)


def merge_train_runtime_configs(
    payloads: Iterable[dict[str, Any]],
    *,
    gpu: str,
    model: str,
) -> dict[str, Any]:
    """Combine multiple anvil-train JSON payloads into one.

    First-seen-wins per (op, m_bucket, n_bucket, k_bucket) cell -- mirrors
    the inference `merge_runtime_configs` semantics. Useful for stitching
    together sweep results from multiple runs (e.g. mxfp4 + int4 separate
    invocations).

    Args:
        payloads: iterable of decoded JSON dicts. Schema mismatch on any
            payload is silently skipped -- callers should validate up front
            if strict checking is required.
        gpu: GPU string for the merged output.
        model: Model name for the merged output.

    Returns:
        A new payload dict (NOT a JSON string) with the same v1 shape.
    """
    merged_ops: dict[str, dict[str, dict[str, Any]]] = {}
    batch = 0
    seq = 0
    rocm_version = ""
    torch_version = ""
    triton_version = ""
    kernel_hash = ""

    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        if payload.get("schema") != SCHEMA_VERSION:
            continue
        # Take the first non-empty version/hash strings encountered. The
        # merged payload represents whichever sweep produced these values
        # most authoritatively -- callers can override after the fact if
        # they have a better source.
        if not rocm_version and payload.get("rocm_version"):
            rocm_version = str(payload["rocm_version"])
        if not torch_version and payload.get("torch_version"):
            torch_version = str(payload["torch_version"])
        if not triton_version and payload.get("triton_version"):
            triton_version = str(payload["triton_version"])
        if not kernel_hash and payload.get("kernel_hash"):
            kernel_hash = str(payload["kernel_hash"])
        if not batch and payload.get("batch"):
            try:
                batch = int(payload["batch"])
            except (TypeError, ValueError):
                pass
        if not seq and payload.get("seq"):
            try:
                seq = int(payload["seq"])
            except (TypeError, ValueError):
                pass

        ops = payload.get("ops") or {}
        if not isinstance(ops, dict):
            continue
        for op_name, cells in ops.items():
            if not isinstance(cells, dict):
                continue
            dest = merged_ops.setdefault(op_name, {})
            for cell_key, cell_payload in cells.items():
                if cell_key in dest:
                    # First-seen wins.
                    continue
                cfg = _coerce_payload(cell_payload) if isinstance(cell_payload, dict) else None
                if cfg is not None:
                    dest[cell_key] = cfg.to_dict()

    return {
        "schema": SCHEMA_VERSION,
        "gpu": gpu,
        "rocm_version": rocm_version,
        "torch_version": torch_version,
        "triton_version": triton_version,
        "kernel_hash": kernel_hash,
        "model": model,
        "batch": batch,
        "seq": seq,
        "ops": merged_ops,
    }
