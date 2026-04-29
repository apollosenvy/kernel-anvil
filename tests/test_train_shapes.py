"""Tests for HF-config -> training-shape extraction.

We DO NOT hit the network. Tests use stand-in config objects whose
attribute names match the transformers config classes (LlamaConfig,
Qwen3Config, GptOssConfig). The extractor walks attribute names rather
than isinstance checks, so a bare object with the right fields is enough
to exercise the shape math.
"""

from __future__ import annotations

import pytest

from kernel_anvil.train_shapes import (
    DEFAULT_OPS,
    extract_shapes,
    model_basename,
)


def _make_config(class_name: str, **fields):
    """Build a stand-in config object whose class name drives dispatch.

    We can't use SimpleNamespace because its constructor stores
    ``__class__`` as a plain attribute -- ``type(obj).__name__`` is still
    'SimpleNamespace'. A dynamically-created class gets us a real
    ``type(obj).__name__`` that the extractor walks.
    """
    cls = type(class_name, (), {})
    obj = cls()
    for k, v in fields.items():
        setattr(obj, k, v)
    return obj


def _qwen3_8b_config():
    return _make_config(
        "Qwen3Config",
        hidden_size=4096,
        intermediate_size=12288,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
    )


def _llama_8b_config():
    return _make_config(
        "LlamaConfig",
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
    )


def _gpt_oss_20b_config():
    """gpt-oss-20b MoE config stand-in."""
    return _make_config(
        "GptOssConfig",
        hidden_size=2880,
        intermediate_size=2880,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
        num_local_experts=32,
        num_experts_per_tok=4,
    )


# ---------------------------------------------------------------------------
# extract_shapes
# ---------------------------------------------------------------------------


class TestExtractShapesQwen3:
    def test_default_ops_emit_all_four(self):
        cfg = _qwen3_8b_config()
        shapes = extract_shapes(cfg, batch=1, seq=4096)
        ops = {op for op, _, _, _ in shapes}
        assert ops == set(DEFAULT_OPS)

    def test_unique_dense_shapes(self):
        cfg = _qwen3_8b_config()
        shapes = extract_shapes(cfg, batch=1, seq=4096, ops=["mxfp4_fwd"])
        # Pull out (M, N, K) for the single op
        mnk = sorted({(M, N, K) for _, M, N, K in shapes})
        # Expected: q_proj/o_proj (4096, 4096), gate/up (4096, 12288, 4096),
        # down (4096, 4096, 12288), kv (4096, 1024, 4096) since
        # 8 kv-heads * 128 head_dim = 1024.
        assert (4096, 4096, 4096) in mnk
        assert (4096, 12288, 4096) in mnk
        assert (4096, 4096, 12288) in mnk
        assert (4096, 1024, 4096) in mnk

    def test_M_scales_with_batch_and_seq(self):
        cfg = _qwen3_8b_config()
        shapes = extract_shapes(cfg, batch=2, seq=2048, ops=["mxfp4_fwd"])
        # M should always be batch*seq = 4096 here; same numeric M as above.
        for _, M, _, _ in shapes:
            assert M == 4096

    def test_dedup(self):
        cfg = _qwen3_8b_config()
        shapes = extract_shapes(cfg, batch=1, seq=4096)
        # Should be deduplicated
        assert len(shapes) == len(set(shapes))


class TestExtractShapesGptOss:
    def test_moe_emits_gate_up_and_down(self):
        cfg = _gpt_oss_20b_config()
        shapes = extract_shapes(cfg, batch=1, seq=2048, ops=["mxfp4_fwd"])
        mnk = {(M, N, K) for _, M, N, K in shapes}
        # gate_up: N = 2 * intermediate = 5760; down: N = hidden = 2880
        assert any(N == 5760 and K == 2880 for _, N, K in mnk) or any(
            n == 5760 and k == 2880 for _, n, k in mnk
        )
        assert any(N == 2880 and K == 2880 for _, N, K in mnk) or any(
            n == 2880 and k == 2880 for _, n, k in mnk
        )

    def test_moe_per_expert_M_smaller_than_dense_M(self):
        cfg = _gpt_oss_20b_config()
        shapes = extract_shapes(cfg, batch=1, seq=2048, ops=["mxfp4_fwd"])
        # Dense M = 2048; per-expert M = ceil(2048 * top_k / num_experts) =
        # ceil(2048 * 4 / 32) = 256.
        ms = {M for _, M, _, _ in shapes}
        assert 256 in ms or any(M < 2048 for M in ms)


class TestExtractShapesValidation:
    def test_empty_ops_rejected(self):
        cfg = _qwen3_8b_config()
        with pytest.raises(ValueError):
            extract_shapes(cfg, batch=1, seq=4096, ops=[])

    def test_zero_batch_rejected(self):
        cfg = _qwen3_8b_config()
        with pytest.raises(ValueError):
            extract_shapes(cfg, batch=0, seq=4096)

    def test_negative_seq_rejected(self):
        cfg = _qwen3_8b_config()
        with pytest.raises(ValueError):
            extract_shapes(cfg, batch=1, seq=-1)

    def test_unknown_arch_rejected(self):
        # No hidden_size, no intermediate_size -- can't synthesize shapes.
        cfg = _make_config("WeirdConfig")
        with pytest.raises(ValueError):
            extract_shapes(cfg, batch=1, seq=128)

    def test_generic_fallback_uses_dense_extractor(self):
        # An arch we don't explicitly recognize but has the right attributes
        # falls back to the dense path -- silent default for new HF configs.
        cfg = _make_config(
            "FutureConfig",
            hidden_size=2048,
            intermediate_size=8192,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=128,
        )
        shapes = extract_shapes(cfg, batch=1, seq=512, ops=["mxfp4_fwd"])
        assert shapes, "fallback path should still produce shapes"


# ---------------------------------------------------------------------------
# model_basename
# ---------------------------------------------------------------------------


class TestModelBasename:
    def test_hf_id(self):
        assert model_basename("Qwen/Qwen3-8B") == "Qwen3-8B"

    def test_local_path(self):
        assert model_basename("/home/x/Models/Qwen3-8B") == "Qwen3-8B"

    def test_trailing_slash(self):
        assert model_basename("/home/x/Models/Qwen3-8B/") == "Qwen3-8B"

    def test_empty(self):
        assert model_basename("") == "model"

    def test_simple_name(self):
        assert model_basename("Qwen3-8B") == "Qwen3-8B"
