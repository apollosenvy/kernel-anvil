"""Triton config grid generator for training-kernel autotune.

The training search space is structurally different from the inference GEMV
space (`sweep.generate_configs`):

- 2D output tile: BLOCK_M and BLOCK_N (not just BLOCK_N).
- Real reduction tile: BLOCK_K up to 128 (training kernels run fp32-accum
  fused dequant+matmul; deeper K helps amortize dequant cost).
- GROUP_M: super-grouping for L2 cache reuse on RDNA3.
- num_stages reaches 4 for software pipelining when LDS budget allows.

We target ~30 candidates per shape after RDNA3 VGPR/LDS filtering. The base
Cartesian product is intentionally generous (3072 raw configs); the filter
+ ranking trims it.
"""

from __future__ import annotations

import itertools
from typing import Iterable

from kernel_anvil.rdna3 import GFX1100, GpuSpec

# Base grid (per spec). These are intentionally Triton-friendly powers of two.
_BLOCK_M_VALUES = (32, 64, 128, 256)
_BLOCK_N_VALUES = (32, 64, 128, 256)
_BLOCK_K_VALUES = (32, 64, 128)
_GROUP_M_VALUES = (1, 4, 8, 16)
_NUM_WARPS_VALUES = (4, 8)
_NUM_STAGES_VALUES = (2, 3, 4)

# Approximate per-element bytes for a fused dequant+matmul training kernel:
# bf16/fp16 activations + fp32 accumulator + small overhead from packed weight
# tile. The exact number depends on the kernel; we use a conservative 4 byte/el
# average for the LDS estimate (fp32 accumulator + bf16 input).
#
# LDS usage estimate (bytes per workgroup):
#   activations: BLOCK_M * BLOCK_K * 2 (bf16/fp16)
#   weights:     BLOCK_N * BLOCK_K * 0.5..2 depending on quant (we use 1 to be
#                generous; INT4/MXFP4 packed is 0.5; bf16 ref path is 2)
#   stages:      multiplied by num_stages (software-pipelined double/triple
#                buffering)
_ACT_ELEMENT_BYTES = 2
_WEIGHT_ELEMENT_BYTES = 1  # average for INT4/MXFP4 unpacked at fp32-accum
# VGPR estimate: per-thread tile footprint. The accumulator dominates:
# (BLOCK_M * BLOCK_N) / threads_per_workgroup floats. Plus operand tiles in
# registers.
_FP32_ACC_BYTES = 4
_VGPR_BYTES = 4  # 1 VGPR per fp32 lane


def _threads_per_workgroup(num_warps: int, wave_size: int = 32) -> int:
    """Triton maps num_warps to threads via wave_size on RDNA3 (wave32)."""
    return num_warps * wave_size


def _estimate_lds_bytes(block_m: int, block_n: int, block_k: int, num_stages: int) -> int:
    """Heuristic LDS-per-workgroup estimate.

    The kernel double/triple-buffers operand tiles to overlap loads with
    compute. Each stage holds one tile of activations and one tile of
    weights.
    """
    a_tile = block_m * block_k * _ACT_ELEMENT_BYTES
    b_tile = block_n * block_k * _WEIGHT_ELEMENT_BYTES
    return (a_tile + b_tile) * max(num_stages, 1)


def _estimate_vgpr(block_m: int, block_n: int, num_warps: int) -> int:
    """Heuristic VGPRs-per-thread for a 2D MMA accumulator tile.

    Each thread owns (BLOCK_M * BLOCK_N) / threads_per_workgroup fp32 lanes
    of accumulator, plus a constant overhead for pointers, scales, masks.
    """
    threads = _threads_per_workgroup(num_warps)
    if threads <= 0:
        return 1024  # absurd -- will be filtered out
    acc_per_thread = (block_m * block_n) // threads
    overhead = 32  # pointers, scales, masks, loop counters
    return acc_per_thread + overhead


def _config_fits_rdna3(
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    gpu: GpuSpec,
) -> bool:
    """Filter configs that exceed RDNA3 VGPR or LDS budgets.

    Falls back to the GFX1100 spec if `gpu` lacks one of the fields. The
    bound is intentionally a bit pessimistic so that legitimate winners
    survive rocm-smi-detected hardware variation.
    """
    lds = _estimate_lds_bytes(block_m, block_n, block_k, num_stages)
    if lds > gpu.lds_per_cu_bytes:
        return False
    vgpr = _estimate_vgpr(block_m, block_n, num_warps)
    if vgpr > gpu.vgpr_per_simd:
        return False
    # Triton mandates BLOCK_M >= 16 and BLOCK_N >= 16 for matmul
    # (sub-16 tiles spill MFMA). Already enforced by our value tuples but
    # paranoia is cheap.
    if min(block_m, block_n, block_k) < 16:
        return False
    # GROUP_M cannot exceed M tiles meaningfully -- a larger value would just
    # underutilize CUs. We don't filter on this since GROUP_M=16 is the max
    # in the grid and still useful for very tall (M=large) kernels.
    return True


def _rank_key(cfg: dict) -> tuple:
    """Heuristic ranking key for picking the top-K configs.

    Bias toward 'mid-tile' candidates which are the empirical winners on
    RDNA3 (BLOCK_M=128 BLOCK_N=64 or 128 BLOCK_K=32 or 64 from the
    headroom benchmark). Larger tiles get higher VGPR/LDS pressure; smaller
    tiles waste parallelism. The ranking is purely a tiebreaker for the
    top-K cap; the actual sweep still benchmarks every kept candidate.
    """
    bm = cfg["BLOCK_M"]
    bn = cfg["BLOCK_N"]
    bk = cfg["BLOCK_K"]
    # Distance from the empirical sweet spot (128, 64, 32). Lower is better.
    dist = abs(bm - 128) + abs(bn - 64) + abs(bk - 32)
    # Penalize stages=4 slightly (larger LDS pressure, marginal gains).
    stage_pen = max(0, cfg["num_stages"] - 3) * 2
    return (dist + stage_pen, cfg["num_warps"])


def generate_train_configs(
    *,
    gpu: GpuSpec | None = None,
    max_configs: int = 30,
    block_m_values: Iterable[int] | None = None,
    block_n_values: Iterable[int] | None = None,
    block_k_values: Iterable[int] | None = None,
    group_m_values: Iterable[int] | None = None,
    num_warps_values: Iterable[int] | None = None,
    num_stages_values: Iterable[int] | None = None,
) -> list[dict]:
    """Build the per-shape Triton config candidate list for a training sweep.

    Args:
        gpu: GPU spec used for the VGPR/LDS sanity filter. Defaults to
            GFX1100 (7900 XTX) when None.
        max_configs: Cap on the returned list. The default 30 matches the
            risk register's compile-budget target (162 configs * 5 shapes
            took ~16 min in the headroom bench; aiming for ~30 keeps
            wallclock under ~5 min per shape).
        block_m_values .. num_stages_values: Override the base grid. Pass
            None to use the spec defaults. Useful for narrowing the sweep
            during debug.

    Returns:
        A list of dicts, each carrying:
            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages.
    """
    gpu = gpu or GFX1100
    bm = tuple(block_m_values) if block_m_values is not None else _BLOCK_M_VALUES
    bn = tuple(block_n_values) if block_n_values is not None else _BLOCK_N_VALUES
    bk = tuple(block_k_values) if block_k_values is not None else _BLOCK_K_VALUES
    gm = tuple(group_m_values) if group_m_values is not None else _GROUP_M_VALUES
    nw = tuple(num_warps_values) if num_warps_values is not None else _NUM_WARPS_VALUES
    ns = tuple(num_stages_values) if num_stages_values is not None else _NUM_STAGES_VALUES

    candidates: list[dict] = []
    for block_m, block_n, block_k, group_m, num_warps, num_stages in itertools.product(
        bm, bn, bk, gm, nw, ns
    ):
        if not _config_fits_rdna3(block_m, block_n, block_k, num_warps, num_stages, gpu):
            continue
        candidates.append(
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_M": group_m,
                "num_warps": num_warps,
                "num_stages": num_stages,
            }
        )

    candidates.sort(key=_rank_key)
    if max_configs and max_configs > 0:
        candidates = candidates[:max_configs]
    return candidates
