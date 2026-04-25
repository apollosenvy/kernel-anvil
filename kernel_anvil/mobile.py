"""Mobile GPU hardware constants and optimization heuristics.

Supports Qualcomm Adreno 7xx/8xx and ARM Mali Valhall (G715/G720/G820).
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MobileGpuSpec:
    name: str
    vendor: str  # "qualcomm" or "arm"
    arch: str  # e.g., "adreno-7xx", "valhall-5th-gen"
    wavefront_size: int
    max_vgprs: int
    lds_size_kb: int
    shader_cores: int
    memory_bandwidth_gbps: float
    max_clock_mhz: int
    fp16_tflops: float
    vulkan_supported: bool = True
    vulkan_min_version: str = "1.1"
    notes: str = ""

    @property
    def lds_size_bytes(self) -> int:
        return self.lds_size_kb * 1024

    def max_vgpr_waves(self, vgpr_count: int) -> int:
        """Max concurrent waves given VGPR usage.

        Adreno allocates VGPRs per fiber (thread), Mali per warp.
        Both architectures cap at a hardware-defined max waves.
        """
        if vgpr_count == 0:
            return self._max_waves
        return min(self._max_waves, self.max_vgprs // vgpr_count)

    def max_lds_waves(self, lds_bytes: int, threads_per_wg: int) -> int:
        """Max concurrent waves given LDS (shared memory) usage."""
        if lds_bytes == 0:
            return self._max_waves
        wgs_per_core = self.lds_size_bytes // max(lds_bytes, 256)
        waves_per_wg = (threads_per_wg + self.wavefront_size - 1) // self.wavefront_size
        total_waves = wgs_per_core * waves_per_wg
        return min(self._max_waves, total_waves)

    def occupancy(self, vgpr_count: int, lds_bytes: int, threads_per_wg: int) -> tuple[float, str]:
        """Returns (occupancy_pct, limiting_factor)."""
        vgpr_w = self.max_vgpr_waves(vgpr_count)
        lds_w = self.max_lds_waves(lds_bytes, threads_per_wg)
        active = min(vgpr_w, lds_w)
        pct = active / self._max_waves * 100
        if vgpr_w < lds_w:
            factor = "vgpr"
        elif lds_w < vgpr_w:
            factor = "lds"
        else:
            factor = "balanced"
        return pct, factor

    @property
    def _max_waves(self) -> int:
        """Hardware max concurrent waves per shader core.

        Adreno 7xx/8xx: up to 16 waves per SP (shader processor).
        Mali Valhall: up to 8 warps per execution engine.
        """
        if self.vendor == "qualcomm":
            return 16
        return 8  # ARM Mali


# Qualcomm Adreno GPUs (Snapdragon SoCs)
# Adreno uses 64-wide wavefronts (fibers), similar to AMD's wave64 mode.
# VGPRs are allocated per-fiber with 256 registers available per SP.
ADRENO_750 = MobileGpuSpec(
    name="Adreno 750", vendor="qualcomm", arch="adreno-7xx",
    wavefront_size=64, max_vgprs=256, lds_size_kb=32,
    shader_cores=6, memory_bandwidth_gbps=77,
    max_clock_mhz=903, fp16_tflops=4.6,
    notes="Snapdragon 8 Gen 3",
)

ADRENO_740 = MobileGpuSpec(
    name="Adreno 740", vendor="qualcomm", arch="adreno-7xx",
    wavefront_size=64, max_vgprs=256, lds_size_kb=32,
    shader_cores=5, memory_bandwidth_gbps=51.2,
    max_clock_mhz=680, fp16_tflops=3.4,
    notes="Snapdragon 8 Gen 2",
)

ADRENO_830 = MobileGpuSpec(
    name="Adreno 830", vendor="qualcomm", arch="adreno-8xx",
    wavefront_size=64, max_vgprs=256, lds_size_kb=32,
    shader_cores=8, memory_bandwidth_gbps=102,
    max_clock_mhz=1000, fp16_tflops=6.0,
    notes="Snapdragon 8 Elite / 8 Gen 4",
)

# ARM Mali GPUs (Exynos, Dimensity, Tensor SoCs)
# Mali Valhall uses 16-wide execution engines (warps).
# 64 registers per warp, shared memory (tile buffer) is 16 KB per core.
MALI_G720 = MobileGpuSpec(
    name="Mali-G720", vendor="arm", arch="valhall-5th-gen",
    wavefront_size=16, max_vgprs=64, lds_size_kb=16,
    shader_cores=12, memory_bandwidth_gbps=51.2,
    max_clock_mhz=1000, fp16_tflops=3.8,
    notes="Dimensity 9300",
)

MALI_G715 = MobileGpuSpec(
    name="Mali-G715", vendor="arm", arch="valhall-4th-gen",
    wavefront_size=16, max_vgprs=64, lds_size_kb=16,
    shader_cores=10, memory_bandwidth_gbps=44.8,
    max_clock_mhz=850, fp16_tflops=2.7,
    notes="Dimensity 9200 / Exynos 2300",
)

MALI_G820 = MobileGpuSpec(
    name="Mali-G820", vendor="arm", arch="valhall-5th-gen",
    wavefront_size=16, max_vgprs=64, lds_size_kb=16,
    shader_cores=14, memory_bandwidth_gbps=68,
    max_clock_mhz=1100, fp16_tflops=4.9,
    notes="Dimensity 9400 / Exynos 2500",
)

MOBILE_GPU_SPECS = {
    # Qualcomm Adreno
    "adreno-750": ADRENO_750,
    "adreno-740": ADRENO_740,
    "adreno-830": ADRENO_830,
    # ARM Mali
    "mali-g720": MALI_G720,
    "mali-g715": MALI_G715,
    "mali-g820": MALI_G820,
}

# Aliases used in plan's test expectations
MOBILE_GPUS = MOBILE_GPU_SPECS


def get_mobile_gpu(name: str) -> Optional[MobileGpuSpec]:
    """Look up a mobile GPU by its canonical name."""
    return MOBILE_GPU_SPECS.get(name)


# Mobile-specific optimization heuristics
MOBILE_HEURISTICS = [
    "Adreno 64-wide wavefronts match AMD wave64 -- same coalescing strategies apply",
    "Mali 16-wide warps need 4x more workgroups for same occupancy as Adreno/AMD",
    "Mobile LPDDR5 bandwidth (50-100 GB/s) is 5-10x less than desktop GDDR6 -- always bandwidth-bound",
    "Shared memory (LDS) is 16-32 KB on mobile vs 64-96 KB desktop -- halve tile sizes",
    "Thermal throttling is the dominant constraint -- sustained throughput << peak",
    "Prefer FP16 compute: mobile GPUs have 2x FP16 rate vs FP32",
    "Workgroup sizes of 64-128 balance occupancy vs register pressure on mobile",
    "Avoid workgroup sizes > 256 on Mali (register file too small)",
    "On Adreno, local_size_x should be a multiple of 64 (wavefront size)",
    "On Mali, local_size_x should be a multiple of 16 (warp size)",
]
