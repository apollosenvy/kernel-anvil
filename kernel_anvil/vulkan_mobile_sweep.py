"""Vulkan mobile sweep for kernel-anvil.

Generates optimized Vulkan compute shader dispatch parameters for mobile GPUs
(Adreno, Mali). Unlike the desktop vulkan_sweep.py which benchmarks live, this
module produces configuration recommendations based on hardware profiles --
actual on-device benchmarking requires an Android device.

The approach:
1. Select a mobile GPU profile (Adreno or Mali)
2. Generate workgroup size / NUM_ROWS candidates respecting mobile constraints
3. Estimate occupancy for each candidate
4. Rank by predicted throughput (occupancy * bandwidth utilization)
5. Emit optimal specialization constants for Vulkan pipeline creation

Vulkan dequant shaders (e.g., dequant_turbo3_0.comp) use:
  - local_size_x (workgroup size, must be multiple of wavefront/warp size)
  - The shader operates on blocks of 32 elements with 256 threads default

Mobile constraints vs desktop:
  - LDS: 16-32 KB vs 64-96 KB (halve tile sizes)
  - Bandwidth: 50-100 GB/s vs 500-960 GB/s (always bandwidth-bound)
  - Thermals: sustained throughput << peak (prefer lower occupancy that sustains)
"""

from __future__ import annotations

from dataclasses import dataclass

from kernel_anvil.mobile import MobileGpuSpec, MOBILE_GPU_SPECS, get_mobile_gpu


@dataclass
class MobileSweepConfig:
    """A candidate Vulkan dispatch configuration for mobile."""
    workgroup_size: int
    num_rows: int
    estimated_occupancy_pct: float
    limiting_factor: str
    bandwidth_utilization: float  # 0.0 - 1.0 estimate
    score: float  # composite ranking score

    @property
    def label(self) -> str:
        return f"wg={self.workgroup_size}_rows={self.num_rows}"


@dataclass
class MobileSweepResult:
    """Result of a mobile Vulkan sweep."""
    gpu: MobileGpuSpec
    configs: list[MobileSweepConfig]
    best: MobileSweepConfig
    quant_type: str


def _estimate_vgpr_usage(workgroup_size: int, num_rows: int) -> int:
    """Estimate VGPR usage for a dequant kernel.

    Dequant kernels are lightweight: a few index variables, the centroid LUT
    (8 floats = 8 registers), norm, loop counter, output. Scale mildly with
    num_rows since each row needs its own accumulation registers.
    """
    base = 16  # index vars, centroids, norm, output
    per_row = 4  # accumulators per row
    return base + per_row * num_rows


def _estimate_lds_usage(workgroup_size: int, num_rows: int) -> int:
    """Estimate LDS usage for a dequant kernel.

    Dequant shaders typically don't use shared memory for the dequantization
    itself (each thread reads its own block directly). LDS is only used if
    there's a reduction step across the workgroup. Estimate conservatively.
    """
    if num_rows > 1:
        # Multi-row may need a small reduction buffer
        return workgroup_size * num_rows * 2  # FP16 = 2 bytes per element
    return 0


def _bandwidth_utilization(gpu: MobileGpuSpec, workgroup_size: int) -> float:
    """Estimate how well the workgroup size utilizes memory bandwidth.

    Optimal when workgroup_size is a multiple of the wavefront/warp size and
    large enough to saturate memory channels, but not so large that it wastes
    register file space.
    """
    wf = gpu.wavefront_size
    # Perfect alignment bonus
    aligned = 1.0 if workgroup_size % wf == 0 else 0.85

    # Size sweet spot: enough waves for latency hiding without over-provisioning
    waves_in_wg = workgroup_size / wf
    if gpu.vendor == "qualcomm":
        # Adreno sweet spot: 2-4 waves per workgroup
        if 2 <= waves_in_wg <= 4:
            size_factor = 1.0
        elif waves_in_wg == 1:
            size_factor = 0.75  # too few for latency hiding
        else:
            size_factor = 0.85  # diminishing returns
    else:
        # Mali sweet spot: 4-8 warps per workgroup (16-wide warps)
        if 4 <= waves_in_wg <= 8:
            size_factor = 1.0
        elif waves_in_wg < 4:
            size_factor = 0.7  # Mali needs more warps for occupancy
        else:
            size_factor = 0.85

    return aligned * size_factor


def generate_mobile_configs(
    gpu: MobileGpuSpec,
    quant_type: str = "turbo3_0",
    max_configs: int = 20,
) -> list[MobileSweepConfig]:
    """Generate Vulkan dispatch configs ranked by predicted throughput.

    Args:
        gpu: Mobile GPU spec to target.
        quant_type: Quantization type (affects block size assumptions).
        max_configs: Maximum number of configs to return.

    Returns:
        List of configs sorted by score (best first).
    """
    wf = gpu.wavefront_size

    # Candidate workgroup sizes: multiples of wavefront size
    if gpu.vendor == "qualcomm":
        wg_candidates = [64, 128, 256]  # Adreno: wave64
    else:
        wg_candidates = [16, 32, 64, 128]  # Mali: warp16

    # NUM_ROWS candidates: how many output rows each workgroup processes
    row_candidates = [1, 2, 4]

    configs: list[MobileSweepConfig] = []
    for wg_size in wg_candidates:
        for num_rows in row_candidates:
            # Skip configs that exceed hardware limits
            vgpr_est = _estimate_vgpr_usage(wg_size, num_rows)
            lds_est = _estimate_lds_usage(wg_size, num_rows)

            if lds_est > gpu.lds_size_bytes:
                continue

            occ_pct, factor = gpu.occupancy(vgpr_est, lds_est, wg_size)
            bw_util = _bandwidth_utilization(gpu, wg_size)

            # Composite score: occupancy * bandwidth utilization
            # Weight occupancy slightly less for mobile (thermal throttling
            # means max occupancy isn't always best)
            score = (occ_pct / 100.0) * 0.6 + bw_util * 0.4

            configs.append(MobileSweepConfig(
                workgroup_size=wg_size,
                num_rows=num_rows,
                estimated_occupancy_pct=occ_pct,
                limiting_factor=factor,
                bandwidth_utilization=bw_util,
                score=score,
            ))

    configs.sort(key=lambda c: c.score, reverse=True)
    return configs[:max_configs]


def sweep_mobile(
    gpu_name: str,
    quant_type: str = "turbo3_0",
    max_configs: int = 20,
) -> MobileSweepResult | None:
    """Run a mobile Vulkan sweep for a named GPU.

    Args:
        gpu_name: Canonical GPU name (e.g., "adreno-750", "mali-g720").
        quant_type: Quantization type to optimize for.
        max_configs: Maximum configs to evaluate.

    Returns:
        MobileSweepResult with ranked configs, or None if GPU not found.
    """
    gpu = get_mobile_gpu(gpu_name)
    if gpu is None:
        return None

    configs = generate_mobile_configs(gpu, quant_type=quant_type, max_configs=max_configs)
    if not configs:
        return None

    return MobileSweepResult(
        gpu=gpu,
        configs=configs,
        best=configs[0],
        quant_type=quant_type,
    )


def sweep_all_mobile(
    quant_type: str = "turbo3_0",
    max_configs: int = 5,
) -> dict[str, MobileSweepResult]:
    """Run sweep for all known mobile GPUs.

    Returns:
        Dict mapping GPU name to sweep result.
    """
    results = {}
    for name in MOBILE_GPU_SPECS:
        result = sweep_mobile(name, quant_type=quant_type, max_configs=max_configs)
        if result is not None:
            results[name] = result
    return results
