"""Tests for model-level auto-optimization (kernel_anvil.model)."""
import json
import os
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from kernel_anvil.model import (
    SmithyLinear,
    _cache_path,
    _find_linears,
    _load_configs,
    _replace_linear,
    _save_configs,
    optimize,
)


# ---------------------------------------------------------------------------
# Helpers: tiny model that looks like a transformer
# ---------------------------------------------------------------------------


class TinyMLP(nn.Module):
    """Minimal model with multiple Linear layers for testing."""

    def __init__(self, hidden=32, intermediate=64):
        super().__init__()
        self.up = nn.Linear(hidden, intermediate)
        self.down = nn.Linear(intermediate, hidden)

    def forward(self, x):
        return self.down(torch.relu(self.up(x)))


class TinyTransformer(nn.Module):
    """Nested model mimicking HF transformer structure."""

    def __init__(self, hidden=32, intermediate=64, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "attn": nn.ModuleDict(
                            {
                                "q_proj": nn.Linear(hidden, hidden, bias=False),
                                "k_proj": nn.Linear(hidden, hidden, bias=False),
                                "v_proj": nn.Linear(hidden, hidden, bias=False),
                                "o_proj": nn.Linear(hidden, hidden, bias=False),
                            }
                        ),
                        "mlp": nn.ModuleDict(
                            {
                                "gate_proj": nn.Linear(hidden, intermediate, bias=False),
                                "up_proj": nn.Linear(hidden, intermediate, bias=False),
                                "down_proj": nn.Linear(intermediate, hidden, bias=False),
                            }
                        ),
                    }
                )
            )
        self.lm_head = nn.Linear(hidden, 100, bias=False)  # vocab size 100

    def forward(self, x):
        for layer in self.layers:
            # Simplified: no real attention, just exercises the linear layers
            q = layer["attn"]["q_proj"](x)
            x = layer["attn"]["o_proj"](q)
            x = layer["mlp"]["down_proj"](torch.relu(layer["mlp"]["gate_proj"](x)))
        return self.lm_head(x)


class FakeConfig:
    """Mimics a HuggingFace PretrainedConfig for cache key generation."""

    def __init__(self, model_type="test", hidden_size=32, num_hidden_layers=2):
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers


# ---------------------------------------------------------------------------
# Tests: SmithyLinear correctness
# ---------------------------------------------------------------------------


class TestSmithyLinearCorrectness:
    """Verify SmithyLinear produces correct output vs nn.Linear (CPU fallback path)."""

    def test_forward_matches_linear_2d(self):
        """SmithyLinear forward (fallback path) matches nn.Linear for M>1."""
        linear = nn.Linear(16, 8)
        sl = SmithyLinear(linear.weight.data.clone(), linear.bias.data.clone(), {})
        x = torch.randn(4, 16)  # batch=4, M>1 -> fallback
        expected = linear(x)
        actual = sl(x)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_forward_matches_linear_3d(self):
        """SmithyLinear forward (fallback path) matches nn.Linear for 3D input."""
        linear = nn.Linear(16, 8, bias=False)
        sl = SmithyLinear(linear.weight.data.clone(), None, {})
        x = torch.randn(2, 3, 16)  # batch=2, seq=3, M=6 -> fallback
        expected = linear(x)
        actual = sl(x)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_forward_no_bias(self):
        """SmithyLinear works without bias (fallback path)."""
        linear = nn.Linear(16, 8, bias=False)
        sl = SmithyLinear(linear.weight.data.clone(), None, {})
        x = torch.randn(2, 16)
        expected = linear(x)
        actual = sl(x)
        assert torch.allclose(actual, expected, atol=1e-6)


class TestSmithyLinearFallback:
    """Verify that SmithyLinear falls back to torch for M>1."""

    def test_batch_input_uses_fallback(self):
        """M>1 should use F.linear, not the Triton kernel."""
        linear = nn.Linear(16, 8)
        sl = SmithyLinear(linear.weight.data.clone(), linear.bias.data.clone(), {})
        x = torch.randn(4, 16)
        with patch("kernel_anvil.model.F.linear", wraps=torch.nn.functional.linear) as mock_linear:
            _ = sl(x)
            mock_linear.assert_called_once()

    def test_3d_batch_uses_fallback(self):
        """[2, 3, hidden] should use fallback since 2*3=6 > 1."""
        linear = nn.Linear(16, 8, bias=False)
        sl = SmithyLinear(linear.weight.data.clone(), None, {})
        x = torch.randn(2, 3, 16)
        with patch("kernel_anvil.model.F.linear", wraps=torch.nn.functional.linear) as mock_linear:
            _ = sl(x)
            mock_linear.assert_called_once()

    def test_single_token_cpu_uses_fallback(self):
        """M=1 on CPU should use F.linear fallback (Triton requires GPU)."""
        linear = nn.Linear(16, 8)
        sl = SmithyLinear(linear.weight.data.clone(), linear.bias.data.clone(), {})
        x = torch.randn(1, 16)  # CPU tensor
        with patch("kernel_anvil.model.F.linear", wraps=torch.nn.functional.linear) as mock_linear:
            _ = sl(x)
            mock_linear.assert_called_once()

    def test_total_tokens_computation(self):
        """Verify _total_tokens for various input shapes."""
        sl = SmithyLinear(torch.randn(8, 16), None, {})
        assert sl._total_tokens(torch.randn(1, 16)) == 1
        assert sl._total_tokens(torch.randn(4, 16)) == 4
        assert sl._total_tokens(torch.randn(1, 1, 16)) == 1
        assert sl._total_tokens(torch.randn(2, 3, 16)) == 6
        assert sl._total_tokens(torch.randn(16)) == 1


# ---------------------------------------------------------------------------
# Tests: _find_linears
# ---------------------------------------------------------------------------


class TestFindLinears:
    def test_finds_all_linears_in_flat_model(self):
        model = TinyMLP(hidden=32, intermediate=64)
        linears = _find_linears(model)
        assert len(linears) == 2
        names = {name for name, _ in linears}
        assert names == {"up", "down"}

    def test_finds_all_linears_in_nested_model(self):
        model = TinyTransformer(hidden=32, intermediate=64, num_layers=2)
        linears = _find_linears(model)
        # 2 layers * (4 attn + 3 mlp) + 1 lm_head = 15
        assert len(linears) == 15

    def test_does_not_find_smithy_linears(self):
        """SmithyLinear should not be found by _find_linears."""
        model = TinyMLP()
        # Manually replace one
        orig = model.up
        model.up = SmithyLinear(orig.weight.data, orig.bias.data, {})
        linears = _find_linears(model)
        # Only 'down' should be found now
        assert len(linears) == 1
        assert linears[0][0] == "down"

    def test_empty_model(self):
        model = nn.Module()
        linears = _find_linears(model)
        assert linears == []


# ---------------------------------------------------------------------------
# Tests: _replace_linear
# ---------------------------------------------------------------------------


class TestReplaceLinear:
    def test_replace_flat(self):
        model = TinyMLP()
        orig = model.up
        replacement = SmithyLinear(orig.weight.data, orig.bias.data, {"BLOCK_N": 64})
        _replace_linear(model, "up", replacement)
        assert isinstance(model.up, SmithyLinear)
        assert model.up.config == {"BLOCK_N": 64}

    def test_replace_nested(self):
        model = TinyTransformer(hidden=32, intermediate=64, num_layers=1)
        name = "layers.0.attn.q_proj"
        orig = model.layers[0]["attn"]["q_proj"]
        replacement = SmithyLinear(orig.weight.data, None, {"BLOCK_N": 32})
        _replace_linear(model, name, replacement)
        assert isinstance(model.layers[0]["attn"]["q_proj"], SmithyLinear)


# ---------------------------------------------------------------------------
# Tests: config caching
# ---------------------------------------------------------------------------


class TestConfigCaching:
    def test_save_and_load(self, tmp_path):
        configs = {"(64, 32)": {"BLOCK_N": 64, "BLOCK_K": 128}}
        path = str(tmp_path / "test_cache.json")
        _save_configs(path, configs)
        loaded = _load_configs(path)
        assert loaded == configs

    def test_load_nonexistent_returns_none(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        assert _load_configs(path) is None

    def test_cache_path_with_hf_config(self, tmp_path):
        model = TinyMLP()
        model.config = FakeConfig(model_type="llama", hidden_size=256, num_hidden_layers=4)
        path = _cache_path(model, str(tmp_path))
        assert path.endswith("llama_256_4.json")
        assert str(tmp_path) in path

    def test_cache_path_without_config(self, tmp_path):
        model = TinyMLP()
        path = _cache_path(model, str(tmp_path))
        assert path.endswith(".json")
        # Should use md5 hash
        basename = os.path.basename(path)
        assert len(basename) > 5  # hash + .json

    def test_cache_dir_created(self, tmp_path):
        subdir = str(tmp_path / "new_subdir" / "cache")
        model = TinyMLP()
        path = _cache_path(model, subdir)
        assert os.path.isdir(subdir)

    def test_round_trip_preserves_types(self, tmp_path):
        """Config values survive JSON round-trip as correct types."""
        configs = {
            "(128, 64)": {"BLOCK_N": 128, "BLOCK_K": 256, "num_warps": 4, "num_stages": 2}
        }
        path = str(tmp_path / "roundtrip.json")
        _save_configs(path, configs)
        loaded = _load_configs(path)
        for shape_key, cfg in loaded.items():
            for k, v in cfg.items():
                assert isinstance(v, int), f"Expected int for {k}, got {type(v)}"


# ---------------------------------------------------------------------------
# Tests: optimize() end-to-end (CPU, no Triton -- tests structure only)
# ---------------------------------------------------------------------------


class TestOptimize:
    def test_replaces_all_linears(self, tmp_path):
        model = TinyMLP(hidden=16, intermediate=32)
        optimize(model, cache_dir=str(tmp_path), verbose=False)
        # Both layers should now be SmithyLinear
        assert isinstance(model.up, SmithyLinear)
        assert isinstance(model.down, SmithyLinear)

    def test_replaces_nested_linears(self, tmp_path):
        model = TinyTransformer(hidden=16, intermediate=32, num_layers=1)
        optimize(model, cache_dir=str(tmp_path), verbose=False)
        # Check a few nested modules
        assert isinstance(model.layers[0]["attn"]["q_proj"], SmithyLinear)
        assert isinstance(model.layers[0]["mlp"]["gate_proj"], SmithyLinear)
        assert isinstance(model.lm_head, SmithyLinear)

    def test_preserves_weight_data(self, tmp_path):
        model = TinyMLP(hidden=16, intermediate=32)
        orig_weight = model.up.weight.data.clone()
        orig_bias = model.up.bias.data.clone()
        optimize(model, cache_dir=str(tmp_path), verbose=False)
        assert torch.equal(model.up.weight.data, orig_weight)
        assert torch.equal(model.up.bias.data, orig_bias)

    def test_preserves_no_bias(self, tmp_path):
        model = TinyTransformer(hidden=16, intermediate=32, num_layers=1)
        optimize(model, cache_dir=str(tmp_path), verbose=False)
        # TinyTransformer uses bias=False everywhere
        assert model.layers[0]["attn"]["q_proj"].bias is None

    def test_saves_cache(self, tmp_path):
        model = TinyMLP(hidden=16, intermediate=32)
        optimize(model, cache_dir=str(tmp_path), verbose=False)
        # Cache file should exist
        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) == 1

    def test_loads_from_cache(self, tmp_path):
        # First run: creates cache
        model1 = TinyMLP(hidden=16, intermediate=32)
        optimize(model1, cache_dir=str(tmp_path), verbose=False)

        # Verify cache was created
        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) == 1

        # Read cache to verify it has content
        with open(cache_files[0]) as f:
            cached = json.load(f)
        assert len(cached) > 0

        # Second run: loads from cache
        model2 = TinyMLP(hidden=16, intermediate=32)
        optimize(model2, cache_dir=str(tmp_path), verbose=False)

        # Both should have SmithyLinear layers
        assert isinstance(model2.up, SmithyLinear)
        assert isinstance(model2.down, SmithyLinear)

    def test_cache_keyed_by_shape(self, tmp_path):
        model = TinyMLP(hidden=16, intermediate=32)
        optimize(model, cache_dir=str(tmp_path), verbose=False)
        cache_files = list(tmp_path.glob("*.json"))
        with open(cache_files[0]) as f:
            cached = json.load(f)
        # TinyMLP has two shapes: (32, 16) for up, (16, 32) for down
        assert "(32, 16)" in cached
        assert "(16, 32)" in cached

    def test_no_linears_returns_model(self, tmp_path):
        model = nn.Sequential(nn.ReLU(), nn.Sigmoid())
        result = optimize(model, cache_dir=str(tmp_path), verbose=False)
        assert result is model

    def test_returns_same_model_instance(self, tmp_path):
        model = TinyMLP(hidden=16, intermediate=32)
        result = optimize(model, cache_dir=str(tmp_path), verbose=False)
        assert result is model

    def test_idempotent_after_optimization(self, tmp_path):
        """Running optimize twice should not change anything the second time."""
        model = TinyMLP(hidden=16, intermediate=32)
        optimize(model, cache_dir=str(tmp_path), verbose=False)
        # Second call should find no nn.Linear (they're all SmithyLinear now)
        linears = _find_linears(model)
        assert len(linears) == 0

    def test_config_in_replacement(self, tmp_path):
        model = TinyMLP(hidden=16, intermediate=32)
        optimize(model, cache_dir=str(tmp_path), verbose=False)
        # Config should be a dict with expected keys
        cfg = model.up.config
        assert isinstance(cfg, dict)
        assert "BLOCK_N" in cfg
        assert "BLOCK_K" in cfg

    def test_optimize_with_hf_config(self, tmp_path):
        """Model with a config attr gets a named cache key."""
        model = TinyMLP(hidden=16, intermediate=32)
        model.config = FakeConfig(model_type="tiny", hidden_size=16, num_hidden_layers=1)
        optimize(model, cache_dir=str(tmp_path), verbose=False)
        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) == 1
        assert "tiny_16_1" in cache_files[0].name


class TestSmithyLinearProperties:
    def test_extra_repr(self):
        w = torch.randn(8, 16)
        sl = SmithyLinear(w, None, {"BLOCK_N": 64, "BLOCK_K": 128})
        rep = sl.extra_repr()
        assert "in_features=16" in rep
        assert "out_features=8" in rep
        assert "bias=False" in rep
        assert "BLOCK_N=64" in rep

    def test_extra_repr_with_bias(self):
        w = torch.randn(8, 16)
        b = torch.randn(8)
        sl = SmithyLinear(w, b, {})
        rep = sl.extra_repr()
        assert "bias=True" in rep

    def test_weight_is_parameter(self):
        w = torch.randn(8, 16)
        sl = SmithyLinear(w, None, {})
        assert isinstance(sl.weight, nn.Parameter)

    def test_bias_is_parameter_when_present(self):
        w = torch.randn(8, 16)
        b = torch.randn(8)
        sl = SmithyLinear(w, b, {})
        assert isinstance(sl.bias, nn.Parameter)

    def test_shape_attributes(self):
        w = torch.randn(8, 16)
        sl = SmithyLinear(w, None, {})
        assert sl.out_features == 8
        assert sl.in_features == 16


# ---------------------------------------------------------------------------
# Tests: GPU (skip if no CUDA/ROCm)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No CUDA/ROCm GPU available"
)
class TestGPUSmithyLinear:
    def test_triton_gemv_correctness_m1(self):
        """Triton GEMV matches torch for M=1 on GPU."""
        N, K = 64, 32
        linear = nn.Linear(K, N, bias=True).cuda().half()
        sl = SmithyLinear(
            linear.weight.data.clone(),
            linear.bias.data.clone(),
            {"BLOCK_N": 32, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1},
        ).cuda()
        x = torch.randn(1, K, device="cuda", dtype=torch.float16)
        expected = linear(x)
        actual = sl(x)
        assert torch.allclose(actual, expected, atol=1e-2, rtol=1e-2), (
            f"max diff: {(actual - expected).abs().max().item()}"
        )

    def test_triton_gemv_correctness_no_bias(self):
        """Triton GEMV matches torch for M=1 without bias."""
        N, K = 128, 64
        linear = nn.Linear(K, N, bias=False).cuda().half()
        sl = SmithyLinear(
            linear.weight.data.clone(),
            None,
            {"BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1},
        ).cuda()
        x = torch.randn(1, K, device="cuda", dtype=torch.float16)
        expected = linear(x)
        actual = sl(x)
        assert torch.allclose(actual, expected, atol=1e-2, rtol=1e-2), (
            f"max diff: {(actual - expected).abs().max().item()}"
        )

    def test_triton_gemv_3d_single_token(self):
        """Triton GEMV works with [1, 1, K] input (HF decode shape)."""
        N, K = 64, 32
        linear = nn.Linear(K, N, bias=True).cuda().half()
        sl = SmithyLinear(
            linear.weight.data.clone(),
            linear.bias.data.clone(),
            {"BLOCK_N": 32, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1},
        ).cuda()
        x = torch.randn(1, 1, K, device="cuda", dtype=torch.float16)
        expected = linear(x)
        actual = sl(x)
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-2, rtol=1e-2)

    def test_fallback_on_gpu_for_batch(self):
        """M>1 should use F.linear even on GPU."""
        N, K = 64, 32
        linear = nn.Linear(K, N).cuda().half()
        sl = SmithyLinear(
            linear.weight.data.clone(),
            linear.bias.data.clone(),
            {"BLOCK_N": 32, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1},
        ).cuda()
        x = torch.randn(4, K, device="cuda", dtype=torch.float16)
        expected = linear(x)
        actual = sl(x)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_optimize_gpu_model(self, tmp_path):
        """Full optimize() on a GPU model produces correct output."""
        model = TinyMLP(hidden=32, intermediate=64).cuda().half()
        x_orig = torch.randn(4, 32, device="cuda", dtype=torch.float16)
        expected = model(x_orig).clone()

        optimize(model, cache_dir=str(tmp_path), verbose=False)

        # Verify batch (M>1) still works via fallback
        actual = model(x_orig)
        assert torch.allclose(actual, expected, atol=1e-2, rtol=1e-2), (
            f"max diff: {(actual - expected).abs().max().item()}"
        )

    def test_optimize_gpu_model_single_token(self, tmp_path):
        """Optimized model produces correct output for single token."""
        model = TinyMLP(hidden=32, intermediate=64).cuda().half()
        x = torch.randn(1, 32, device="cuda", dtype=torch.float16)

        # Get reference output BEFORE optimization
        with torch.no_grad():
            ref = model(x).clone()

        optimize(model, cache_dir=str(tmp_path), verbose=False)

        with torch.no_grad():
            actual = model(x)
        assert torch.allclose(actual, ref, atol=1e-1, rtol=1e-1), (
            f"max diff: {(actual - ref).abs().max().item()}"
        )

    def test_optimized_larger_shapes(self, tmp_path):
        """Test with shapes more typical of real models."""
        N, K = 256, 128
        linear = nn.Linear(K, N, bias=False).cuda().half()
        x = torch.randn(1, K, device="cuda", dtype=torch.float16)
        expected = linear(x)

        sl = SmithyLinear(
            linear.weight.data.clone(),
            None,
            {"BLOCK_N": 64, "BLOCK_K": 128, "num_warps": 4, "num_stages": 1},
        ).cuda()
        actual = sl(x)
        assert torch.allclose(actual, expected, atol=1e-2, rtol=1e-2)
