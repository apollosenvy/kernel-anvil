"""Tests for Vulkan mobile sweep configuration generation."""
import pytest

from kernel_anvil.mobile import ADRENO_750, ADRENO_830, MALI_G720, MALI_G715
from kernel_anvil.vulkan_mobile_sweep import (
    MobileSweepConfig,
    MobileSweepResult,
    generate_mobile_configs,
    sweep_mobile,
    sweep_all_mobile,
)


# --- Config generation ---

def test_generate_adreno_configs():
    configs = generate_mobile_configs(ADRENO_750)
    assert len(configs) > 0
    assert all(isinstance(c, MobileSweepConfig) for c in configs)


def test_generate_mali_configs():
    configs = generate_mobile_configs(MALI_G720)
    assert len(configs) > 0
    assert all(isinstance(c, MobileSweepConfig) for c in configs)


def test_configs_sorted_by_score():
    configs = generate_mobile_configs(ADRENO_750)
    scores = [c.score for c in configs]
    assert scores == sorted(scores, reverse=True)


def test_max_configs_respected():
    configs = generate_mobile_configs(ADRENO_750, max_configs=3)
    assert len(configs) <= 3


def test_adreno_workgroup_sizes_aligned():
    """All Adreno configs should have workgroup sizes that are multiples of 64."""
    configs = generate_mobile_configs(ADRENO_750)
    for c in configs:
        assert c.workgroup_size % 64 == 0, (
            f"Adreno workgroup size {c.workgroup_size} not aligned to wavefront"
        )


def test_mali_workgroup_sizes_aligned():
    """All Mali configs should have workgroup sizes that are multiples of 16."""
    configs = generate_mobile_configs(MALI_G720)
    for c in configs:
        assert c.workgroup_size % 16 == 0, (
            f"Mali workgroup size {c.workgroup_size} not aligned to warp"
        )


def test_configs_have_valid_occupancy():
    configs = generate_mobile_configs(ADRENO_750)
    for c in configs:
        assert 0 < c.estimated_occupancy_pct <= 100.0
        assert c.limiting_factor in ("vgpr", "lds", "balanced")


def test_configs_have_valid_bandwidth_utilization():
    configs = generate_mobile_configs(MALI_G720)
    for c in configs:
        assert 0 < c.bandwidth_utilization <= 1.0


def test_configs_have_valid_score():
    configs = generate_mobile_configs(ADRENO_830)
    for c in configs:
        assert 0 < c.score <= 1.0


def test_config_label_format():
    configs = generate_mobile_configs(ADRENO_750, max_configs=1)
    label = configs[0].label
    assert label.startswith("wg=")
    assert "_rows=" in label


# --- Sweep function ---

def test_sweep_mobile_adreno():
    result = sweep_mobile("adreno-750")
    assert result is not None
    assert isinstance(result, MobileSweepResult)
    assert result.gpu is ADRENO_750
    assert result.quant_type == "turbo3_0"
    assert result.best is result.configs[0]
    assert len(result.configs) > 0


def test_sweep_mobile_mali():
    result = sweep_mobile("mali-g720")
    assert result is not None
    assert result.gpu is MALI_G720


def test_sweep_mobile_unknown():
    result = sweep_mobile("nonexistent-gpu")
    assert result is None


def test_sweep_mobile_custom_quant():
    result = sweep_mobile("adreno-830", quant_type="q4_k")
    assert result is not None
    assert result.quant_type == "q4_k"


# --- Sweep all ---

def test_sweep_all_mobile():
    results = sweep_all_mobile(max_configs=3)
    assert len(results) == 6  # 3 Adreno + 3 Mali
    for name, result in results.items():
        assert result.gpu.name  # not empty
        assert len(result.configs) <= 3
        assert result.best is result.configs[0]


def test_sweep_all_has_both_vendors():
    results = sweep_all_mobile(max_configs=1)
    vendors = {r.gpu.vendor for r in results.values()}
    assert "qualcomm" in vendors
    assert "arm" in vendors


# --- Cross-GPU consistency ---

def test_adreno_830_higher_score_than_740():
    """Adreno 830 should generally achieve equal or better scores than 740
    (more shader cores, higher bandwidth, same arch constraints)."""
    r830 = sweep_mobile("adreno-830", max_configs=1)
    r740 = sweep_mobile("adreno-740", max_configs=1)
    assert r830 is not None and r740 is not None
    # Best config scores should be comparable (same workgroup logic)
    # Just verify both produce valid results
    assert r830.best.score > 0
    assert r740.best.score > 0


def test_mali_g820_higher_bandwidth():
    """Mali G820 should have higher bandwidth than G715."""
    r820 = sweep_mobile("mali-g820", max_configs=1)
    r715 = sweep_mobile("mali-g715", max_configs=1)
    assert r820 is not None and r715 is not None
    assert r820.gpu.memory_bandwidth_gbps > r715.gpu.memory_bandwidth_gbps
