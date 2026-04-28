"""Tests for the GGUF parser module."""

import io
import os
import sys
from pathlib import Path

import pytest

from kernel_anvil.gguf import ModelProfile, TensorInfo, parse_gguf, print_model_summary

# Override with KERNEL_ANVIL_TEST_GGUF=/path/to/any.gguf to point the
# integration tests at a different model. Tests skip cleanly when the
# file is absent.
QWEN3_PATH = Path(
    os.environ.get(
        "KERNEL_ANVIL_TEST_GGUF",
        str(Path.home() / "Models" / "Qwen3-8B" / "Qwen3-8B-Q4_K_M.gguf"),
    )
)
_skip_no_model = pytest.mark.skipif(
    not QWEN3_PATH.exists(), reason=f"GGUF file not found: {QWEN3_PATH}"
)


# ---------------------------------------------------------------------------
# Unit tests (no model file needed)
# ---------------------------------------------------------------------------


def test_tensor_info_fields():
    t = TensorInfo(name="blk.0.attn_q.weight", shape=(4096, 4096), quant_type="Q4_K", size_bytes=9437184)
    assert t.name == "blk.0.attn_q.weight"
    assert t.shape == (4096, 4096)
    assert t.quant_type == "Q4_K"
    assert t.size_bytes == 9437184


def test_model_profile_fields():
    p = ModelProfile(name="test", architecture="llama", tensors=[], unique_shapes={})
    assert p.name == "test"
    assert p.architecture == "llama"
    assert p.tensors == []
    assert p.unique_shapes == {}


def test_parse_missing_file():
    with pytest.raises(FileNotFoundError):
        parse_gguf("/nonexistent/model.gguf")


def test_print_model_summary_no_crash():
    """print_model_summary should work on a synthetic profile."""
    profile = ModelProfile(
        name="TestModel",
        architecture="llama",
        tensors=[
            TensorInfo("w1", (4096, 4096), "Q4_K", 9437184),
            TensorInfo("norm", (4096,), "F32", 16384),
        ],
        unique_shapes={("Q4_K", 4096, 4096): 1},
    )
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        print_model_summary(profile)
    finally:
        sys.stdout = old_stdout
    output = buf.getvalue()
    assert "TestModel" in output
    assert "llama" in output
    assert "Q4_K" in output


# ---------------------------------------------------------------------------
# Integration tests (require real GGUF file)
# ---------------------------------------------------------------------------


@_skip_no_model
class TestQwen3Parse:
    """Tests against the real Qwen3-8B-Q4_K_M.gguf file."""

    @pytest.fixture(scope="class")
    def profile(self) -> ModelProfile:
        return parse_gguf(QWEN3_PATH)

    def test_model_name(self, profile: ModelProfile):
        assert "Qwen3" in profile.name

    def test_architecture(self, profile: ModelProfile):
        assert profile.architecture == "qwen3"

    def test_tensor_count(self, profile: ModelProfile):
        assert len(profile.tensors) == 399

    def test_2d_tensor_count(self, profile: ModelProfile):
        n2d = sum(1 for t in profile.tensors if len(t.shape) == 2)
        assert n2d == 254

    def test_expected_dimensions(self, profile: ModelProfile):
        """Qwen3-8B should have tensors with these dimensions."""
        all_dims = set()
        for t in profile.tensors:
            for d in t.shape:
                all_dims.add(d)
        # hidden_size
        assert 4096 in all_dims
        # intermediate_size
        assert 12288 in all_dims
        # num_kv_heads * head_dim = 8 * 128 = 1024
        assert 1024 in all_dims
        # vocab_size
        assert 151936 in all_dims

    def test_quant_types(self, profile: ModelProfile):
        """Q4_K_M model should be mostly Q4_K with some Q6_K."""
        quant_types = {qt for (qt, _, _) in profile.unique_shapes}
        assert "Q4_K" in quant_types
        assert "Q6_K" in quant_types

    def test_q4k_dominates(self, profile: ModelProfile):
        """Q4_K should have more 2D tensors than Q6_K."""
        q4k_count = sum(c for (qt, _, _), c in profile.unique_shapes.items() if qt == "Q4_K")
        q6k_count = sum(c for (qt, _, _), c in profile.unique_shapes.items() if qt == "Q6_K")
        assert q4k_count > q6k_count

    def test_unique_shapes_has_expected_workloads(self, profile: ModelProfile):
        """Check key GEMV workloads are present."""
        shapes = profile.unique_shapes
        # Q/V projections: (4096, 1024)
        assert ("Q4_K", 4096, 1024) in shapes
        # Attention output: (4096, 4096)
        assert ("Q4_K", 4096, 4096) in shapes
        # MLP gate/up: (12288, 4096)
        assert ("Q4_K", 12288, 4096) in shapes
        # MLP down: (4096, 12288)
        assert ("Q4_K", 4096, 12288) in shapes

    def test_layer_count_consistency(self, profile: ModelProfile):
        """36 layers means Q/K projections appear 36*1 = 36 times per type,
        but there are Q and K so 36+18=54 for Q4_K (4096,1024)."""
        # Q4_K (4096, 1024) covers attn_q (36 layers) + attn_k Q4_K subset
        count = profile.unique_shapes.get(("Q4_K", 4096, 1024), 0)
        # Should be some multiple related to 36 layers
        assert count > 0
        assert count % 18 == 0  # comes in multiples of 18 (half the layers per quant type)

    def test_print_summary_real(self, profile: ModelProfile):
        """print_model_summary should not crash on real data."""
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            print_model_summary(profile)
        finally:
            sys.stdout = old_stdout
        output = buf.getvalue()
        assert "Qwen3" in output
        assert "qwen3" in output
        assert "254 weight" in output

    def test_total_size_reasonable(self, profile: ModelProfile):
        """Total tensor bytes should be in the right ballpark for a Q4 8B model."""
        total = sum(t.size_bytes for t in profile.tensors)
        # Should be roughly 4-5 GB for a Q4_K_M 8B model
        assert 3e9 < total < 6e9

    def test_tensor_names_follow_pattern(self, profile: ModelProfile):
        """Tensor names should follow the blk.N.* pattern for layer tensors."""
        block_tensors = [t for t in profile.tensors if t.name.startswith("blk.")]
        assert len(block_tensors) > 300  # 36 layers * ~10 tensors each
        # Check layer indices span 0..35
        layer_ids = set()
        for t in block_tensors:
            parts = t.name.split(".")
            layer_ids.add(int(parts[1]))
        assert layer_ids == set(range(36))
