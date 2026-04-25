"""Tests for mobile GPU hardware constants and occupancy calculations."""
import pytest

from kernel_anvil.mobile import (
    ADRENO_750,
    ADRENO_740,
    ADRENO_830,
    MALI_G720,
    MALI_G715,
    MALI_G820,
    MOBILE_GPU_SPECS,
    MOBILE_GPUS,
    MobileGpuSpec,
    get_mobile_gpu,
)


# --- Lookup tests ---

def test_get_adreno_750():
    gpu = get_mobile_gpu("adreno-750")
    assert gpu is not None
    assert gpu is ADRENO_750
    assert gpu.vendor == "qualcomm"
    assert gpu.wavefront_size == 64


def test_get_mali_g720():
    gpu = get_mobile_gpu("mali-g720")
    assert gpu is not None
    assert gpu is MALI_G720
    assert gpu.vendor == "arm"
    assert gpu.wavefront_size == 16


def test_get_unknown_gpu():
    gpu = get_mobile_gpu("nonexistent-gpu")
    assert gpu is None


def test_mobile_gpus_alias():
    """MOBILE_GPUS is the same dict as MOBILE_GPU_SPECS."""
    assert MOBILE_GPUS is MOBILE_GPU_SPECS


# --- Spec validation ---

def test_all_mobile_gpus_have_vulkan():
    for name, gpu in MOBILE_GPU_SPECS.items():
        assert gpu.vulkan_supported, f"{name} must support Vulkan"


def test_all_specs_have_positive_bandwidth():
    for name, gpu in MOBILE_GPU_SPECS.items():
        assert gpu.memory_bandwidth_gbps > 0, f"{name} bandwidth must be positive"
        assert gpu.max_vgprs > 0, f"{name} max_vgprs must be positive"
        assert gpu.shader_cores > 0, f"{name} shader_cores must be positive"
        assert gpu.fp16_tflops > 0, f"{name} fp16_tflops must be positive"


def test_adreno_wavefront_size_64():
    for name, gpu in MOBILE_GPU_SPECS.items():
        if gpu.vendor == "qualcomm":
            assert gpu.wavefront_size == 64, f"{name} should have wavefront_size=64"


def test_mali_wavefront_size_16():
    for name, gpu in MOBILE_GPU_SPECS.items():
        if gpu.vendor == "arm":
            assert gpu.wavefront_size == 16, f"{name} should have wavefront_size=16"


def test_gpu_spec_frozen():
    with pytest.raises(AttributeError):
        ADRENO_750.shader_cores = 8  # type: ignore


def test_lds_size_bytes():
    assert ADRENO_750.lds_size_bytes == 32 * 1024
    assert MALI_G720.lds_size_bytes == 16 * 1024


# --- Occupancy calculations ---

def test_adreno_max_vgpr_waves_zero():
    """Zero VGPRs means max waves."""
    assert ADRENO_750.max_vgpr_waves(0) == 16


def test_adreno_max_vgpr_waves_moderate():
    # 128 VGPRs -> 256/128 = 2
    assert ADRENO_750.max_vgpr_waves(128) == 2


def test_adreno_max_vgpr_waves_full():
    # 256 VGPRs -> 256/256 = 1
    assert ADRENO_750.max_vgpr_waves(256) == 1


def test_mali_max_vgpr_waves_zero():
    """Zero VGPRs means max waves."""
    assert MALI_G720.max_vgpr_waves(0) == 8


def test_mali_max_vgpr_waves_moderate():
    # 32 VGPRs -> 64/32 = 2
    assert MALI_G720.max_vgpr_waves(32) == 2


def test_mali_max_vgpr_waves_full():
    # 64 VGPRs -> 64/64 = 1
    assert MALI_G720.max_vgpr_waves(64) == 1


def test_adreno_max_lds_waves_zero():
    """Zero LDS means max waves."""
    assert ADRENO_750.max_lds_waves(0, 256) == 16


def test_adreno_max_lds_waves_moderate():
    # 8192 LDS bytes, 128 threads
    # wgs_per_core = 32768 / 8192 = 4
    # waves_per_wg = 128 / 64 = 2
    # total_waves = 4 * 2 = 8
    # capped at 8 (< max 16)
    assert ADRENO_750.max_lds_waves(8192, 128) == 8


def test_adreno_max_lds_waves_high():
    # 32768 LDS bytes = all of it, 64 threads
    # wgs_per_core = 32768 / 32768 = 1
    # waves_per_wg = 64 / 64 = 1
    # total_waves = 1 * 1 = 1
    assert ADRENO_750.max_lds_waves(32768, 64) == 1


def test_mali_max_lds_waves_moderate():
    # 4096 LDS bytes, 64 threads
    # wgs_per_core = 16384 / 4096 = 4
    # waves_per_wg = 64 / 16 = 4
    # total_waves = 4 * 4 = 16
    # capped at 8
    assert MALI_G720.max_lds_waves(4096, 64) == 8


def test_occupancy_vgpr_limited():
    # Adreno 750: 256 VGPRs used -> 1 wave, no LDS -> 16 waves
    pct, factor = ADRENO_750.occupancy(vgpr_count=256, lds_bytes=0, threads_per_wg=64)
    assert pct == pytest.approx(6.25)  # 1/16 * 100
    assert factor == "vgpr"


def test_occupancy_lds_limited():
    # Adreno 750: 0 VGPRs -> 16 waves, 32 KB LDS + 64 threads -> 1 wave
    pct, factor = ADRENO_750.occupancy(vgpr_count=0, lds_bytes=32768, threads_per_wg=64)
    assert pct == pytest.approx(6.25)  # 1/16 * 100
    assert factor == "lds"


def test_occupancy_balanced():
    pct, factor = ADRENO_750.occupancy(vgpr_count=0, lds_bytes=0, threads_per_wg=64)
    assert pct == 100.0
    assert factor == "balanced"


def test_mali_occupancy_vgpr_limited():
    # Mali G720: 64 VGPRs -> 1 wave, no LDS -> 8 waves
    pct, factor = MALI_G720.occupancy(vgpr_count=64, lds_bytes=0, threads_per_wg=16)
    assert pct == pytest.approx(12.5)  # 1/8 * 100
    assert factor == "vgpr"


# --- Bandwidth ordering ---

def test_adreno_bandwidth_ordering():
    assert ADRENO_740.memory_bandwidth_gbps < ADRENO_750.memory_bandwidth_gbps < ADRENO_830.memory_bandwidth_gbps


def test_mali_bandwidth_ordering():
    assert MALI_G715.memory_bandwidth_gbps < MALI_G720.memory_bandwidth_gbps < MALI_G820.memory_bandwidth_gbps


def test_spec_count():
    """We should have 6 mobile GPU specs (3 Adreno + 3 Mali)."""
    assert len(MOBILE_GPU_SPECS) == 6
    qualcomm = [g for g in MOBILE_GPU_SPECS.values() if g.vendor == "qualcomm"]
    arm = [g for g in MOBILE_GPU_SPECS.values() if g.vendor == "arm"]
    assert len(qualcomm) == 3
    assert len(arm) == 3
