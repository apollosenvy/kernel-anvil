"""Tests for the anvil-train v1 JSON code generator."""

from __future__ import annotations

import json

import pytest

from kernel_anvil.train_codegen import (
    BUCKET_BOUNDARIES,
    KNOWN_OPS,
    NUM_BUCKETS,
    SCHEMA_VERSION,
    TrainShapeConfig,
    bucket_index_3d,
    build_op_table,
    generate_train_runtime_config,
    merge_train_runtime_configs,
)


# ---------------------------------------------------------------------------
# bucket_index_3d
# ---------------------------------------------------------------------------


class TestBucketIndex3D:
    def test_below_first_boundary_in_all_axes(self):
        assert bucket_index_3d(1, 1, 1) == (0, 0, 0)
        assert bucket_index_3d(1024, 1024, 1024) == (0, 0, 0)

    def test_exact_boundaries_per_axis(self):
        # Each axis uses identical boundaries; bucket i is reached at
        # value == BUCKET_BOUNDARIES[i].
        for i, boundary in enumerate(BUCKET_BOUNDARIES):
            assert bucket_index_3d(boundary, boundary, boundary) == (i, i, i)

    def test_just_above_boundaries(self):
        assert bucket_index_3d(1025, 1, 1) == (1, 0, 0)
        assert bucket_index_3d(1, 4097, 1) == (0, 2, 0)
        assert bucket_index_3d(1, 1, 16385) == (0, 0, 4)

    def test_overflow_to_last_bucket(self):
        big = BUCKET_BOUNDARIES[-1] * 100
        assert bucket_index_3d(big, big, big) == (
            len(BUCKET_BOUNDARIES),
            len(BUCKET_BOUNDARIES),
            len(BUCKET_BOUNDARIES),
        )

    def test_independent_axes(self):
        # Boundaries are (1024, 4096, 8192, 16384). 512 -> bucket 0,
        # 8192 (== boundary index 2) -> bucket 2.
        assert bucket_index_3d(8192, 512, 8192) == (2, 0, 2)


class TestNumBuckets:
    def test_num_buckets_matches_boundary_count_plus_one(self):
        assert NUM_BUCKETS == len(BUCKET_BOUNDARIES) + 1


# ---------------------------------------------------------------------------
# TrainShapeConfig
# ---------------------------------------------------------------------------


class TestTrainShapeConfig:
    def test_required_fields_only(self):
        cfg = TrainShapeConfig(
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32,
            GROUP_M=8, num_warps=8, num_stages=4,
        )
        d = cfg.to_dict()
        assert d == {
            "BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32,
            "GROUP_M": 8, "num_warps": 8, "num_stages": 4,
        }

    def test_optional_fields_round_trip(self):
        cfg = TrainShapeConfig(
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32,
            GROUP_M=8, num_warps=8, num_stages=4,
            speedup_vs_baseline=1.42, profiled_us=18.4,
        )
        d = cfg.to_dict()
        assert d["speedup_vs_baseline"] == 1.42
        assert d["profiled_us"] == 18.4

    def test_frozen(self):
        cfg = TrainShapeConfig(
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32,
            GROUP_M=8, num_warps=8, num_stages=4,
        )
        with pytest.raises(AttributeError):
            cfg.BLOCK_M = 256


# ---------------------------------------------------------------------------
# build_op_table
# ---------------------------------------------------------------------------


class TestBuildOpTable:
    def test_empty(self):
        assert build_op_table({}) == {}

    def test_single_entry_lands_in_correct_cell(self):
        configs = {
            ("mxfp4_fwd", 4096, 4096, 4096): {
                "BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32,
                "GROUP_M": 8, "num_warps": 8, "num_stages": 4,
            },
        }
        table = build_op_table(configs)
        assert "mxfp4_fwd" in table
        # 4096 -> bucket 1 on each axis (boundary == 4096)
        cell_key = "1,1,1"
        assert cell_key in table["mxfp4_fwd"]
        assert table["mxfp4_fwd"][cell_key]["BLOCK_M"] == 128

    def test_multiple_ops_isolated(self):
        configs = {
            ("mxfp4_fwd", 4096, 4096, 4096): {
                "BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32,
                "GROUP_M": 8, "num_warps": 8, "num_stages": 4,
            },
            ("int4_fwd", 4096, 4096, 4096): {
                "BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64,
                "GROUP_M": 4, "num_warps": 4, "num_stages": 2,
            },
        }
        table = build_op_table(configs)
        assert table["mxfp4_fwd"]["1,1,1"]["BLOCK_M"] == 128
        assert table["int4_fwd"]["1,1,1"]["BLOCK_M"] == 64

    def test_first_seen_wins_without_priority(self):
        configs = {
            ("mxfp4_fwd", 3000, 3000, 3000): {
                "BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32,
                "GROUP_M": 8, "num_warps": 8, "num_stages": 4,
            },
            ("mxfp4_fwd", 4000, 4000, 4000): {
                "BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64,
                "GROUP_M": 4, "num_warps": 4, "num_stages": 2,
            },
        }
        table = build_op_table(configs)
        # Both shapes map to bucket cell 1,1,1; first inserted wins.
        assert table["mxfp4_fwd"]["1,1,1"]["BLOCK_M"] == 128

    def test_priority_picks_higher(self):
        configs = {
            ("mxfp4_fwd", 3000, 3000, 3000): {
                "BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32,
                "GROUP_M": 8, "num_warps": 8, "num_stages": 4,
            },
            ("mxfp4_fwd", 4000, 4000, 4000): {
                "BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64,
                "GROUP_M": 4, "num_warps": 4, "num_stages": 2,
            },
        }
        priorities = {
            ("mxfp4_fwd", 3000, 3000, 3000): 100,
            ("mxfp4_fwd", 4000, 4000, 4000): 200,
        }
        table = build_op_table(configs, priorities=priorities)
        # Higher-priority entry wins.
        assert table["mxfp4_fwd"]["1,1,1"]["BLOCK_M"] == 64

    def test_malformed_payload_silently_skipped(self):
        configs = {
            ("mxfp4_fwd", 4096, 4096, 4096): {
                "BLOCK_M": 128,  # missing other required fields
            },
        }
        table = build_op_table(configs)
        assert table == {}

    def test_non_tuple_key_silently_skipped(self):
        configs = {
            "not a tuple": {
                "BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32,
                "GROUP_M": 8, "num_warps": 8, "num_stages": 4,
            },
        }
        table = build_op_table(configs)
        assert table == {}

    def test_zero_or_negative_dims_skipped(self):
        configs = {
            ("mxfp4_fwd", 0, 4096, 4096): {
                "BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32,
                "GROUP_M": 8, "num_warps": 8, "num_stages": 4,
            },
            ("mxfp4_fwd", 4096, -1, 4096): {
                "BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32,
                "GROUP_M": 8, "num_warps": 8, "num_stages": 4,
            },
        }
        table = build_op_table(configs)
        assert table == {}


# ---------------------------------------------------------------------------
# generate_train_runtime_config
# ---------------------------------------------------------------------------


class TestGenerateTrainRuntimeConfig:
    def _sample_configs(self):
        return {
            ("mxfp4_fwd", 4096, 4096, 4096): {
                "BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32,
                "GROUP_M": 8, "num_warps": 8, "num_stages": 4,
                "speedup_vs_baseline": 1.33, "profiled_us": 18.4,
            },
            ("int4_fwd", 8192, 4096, 4096): {
                "BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64,
                "GROUP_M": 4, "num_warps": 4, "num_stages": 3,
            },
        }

    def test_top_level_fields(self):
        text = generate_train_runtime_config(
            self._sample_configs(),
            gpu="gfx1100",
            model="Qwen3-8B",
            batch=1,
            seq=4096,
            rocm_version="7.1",
            torch_version="2.10.0+rocm7.1",
            triton_version="3.6.0",
            kernel_hash="sha256:cafe",
        )
        payload = json.loads(text)
        assert payload["schema"] == SCHEMA_VERSION
        assert payload["gpu"] == "gfx1100"
        assert payload["model"] == "Qwen3-8B"
        assert payload["batch"] == 1
        assert payload["seq"] == 4096
        assert payload["rocm_version"] == "7.1"
        assert payload["torch_version"] == "2.10.0+rocm7.1"
        assert payload["triton_version"] == "3.6.0"
        assert payload["kernel_hash"] == "sha256:cafe"

    def test_ops_keyed_correctly(self):
        text = generate_train_runtime_config(
            self._sample_configs(),
            gpu="gfx1100", model="Qwen3-8B", batch=1, seq=4096,
        )
        payload = json.loads(text)
        assert "mxfp4_fwd" in payload["ops"]
        assert "int4_fwd" in payload["ops"]
        # 4096 -> bucket 1; 8192 -> bucket 2 (boundaries 1024, 4096, 8192, 16384)
        assert "1,1,1" in payload["ops"]["mxfp4_fwd"]
        assert "2,1,1" in payload["ops"]["int4_fwd"]

    def test_optional_fields_round_trip(self):
        text = generate_train_runtime_config(
            self._sample_configs(),
            gpu="gfx1100", model="Qwen3-8B", batch=1, seq=4096,
        )
        payload = json.loads(text)
        cell = payload["ops"]["mxfp4_fwd"]["1,1,1"]
        assert cell["speedup_vs_baseline"] == pytest.approx(1.33)
        assert cell["profiled_us"] == pytest.approx(18.4)

    def test_known_ops_constant(self):
        # Sanity: the contract calls out exactly four well-known ops.
        assert KNOWN_OPS == (
            "mxfp4_fwd",
            "mxfp4_grad_input",
            "int4_fwd",
            "int4_grad_input",
        )


# ---------------------------------------------------------------------------
# merge_train_runtime_configs
# ---------------------------------------------------------------------------


class TestMergeTrainRuntimeConfigs:
    def _payload(self, *cells):
        ops: dict = {}
        for op, cell, body in cells:
            ops.setdefault(op, {})[cell] = body
        return {
            "schema": SCHEMA_VERSION,
            "gpu": "gfx1100",
            "model": "test",
            "batch": 1,
            "seq": 4096,
            "rocm_version": "7.1",
            "torch_version": "2.10.0",
            "triton_version": "3.6.0",
            "kernel_hash": "sha256:abc",
            "ops": ops,
        }

    def _body(self):
        return {
            "BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32,
            "GROUP_M": 8, "num_warps": 8, "num_stages": 4,
        }

    def test_empty_inputs(self):
        merged = merge_train_runtime_configs([], gpu="gfx1100", model="x")
        assert merged["schema"] == SCHEMA_VERSION
        assert merged["ops"] == {}

    def test_disjoint_cells_unioned(self):
        a = self._payload(("mxfp4_fwd", "1,1,1", self._body()))
        b = self._payload(("mxfp4_fwd", "2,2,2", self._body()))
        merged = merge_train_runtime_configs([a, b], gpu="gfx1100", model="m")
        assert "1,1,1" in merged["ops"]["mxfp4_fwd"]
        assert "2,2,2" in merged["ops"]["mxfp4_fwd"]

    def test_first_seen_wins_on_collision(self):
        win = dict(self._body(), BLOCK_M=128)
        lose = dict(self._body(), BLOCK_M=64)
        a = self._payload(("mxfp4_fwd", "1,1,1", win))
        b = self._payload(("mxfp4_fwd", "1,1,1", lose))
        merged = merge_train_runtime_configs([a, b], gpu="gfx1100", model="m")
        assert merged["ops"]["mxfp4_fwd"]["1,1,1"]["BLOCK_M"] == 128

    def test_skips_wrong_schema(self):
        # An "anvil-train/v999" payload must be ignored entirely.
        a = self._payload(("mxfp4_fwd", "1,1,1", self._body()))
        a["schema"] = "anvil-train/v999"
        merged = merge_train_runtime_configs([a], gpu="gfx1100", model="m")
        assert merged["ops"] == {}

    def test_skips_non_dict_payload(self):
        merged = merge_train_runtime_configs(
            [None, "junk", 42], gpu="gfx1100", model="m",
        )
        assert merged["ops"] == {}

    def test_skips_malformed_cells(self):
        a = self._payload(("mxfp4_fwd", "1,1,1", "not a dict"))
        merged = merge_train_runtime_configs([a], gpu="gfx1100", model="m")
        assert merged["ops"].get("mxfp4_fwd", {}) == {}
