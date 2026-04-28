"""Tests for the C header code generation module."""

import sys

import pytest

from kernel_anvil.codegen import (
    BUCKET_BOUNDARIES,
    DEFAULT_CONFIG,
    GGML_TYPE_COUNT,
    GGML_TYPE_MAP,
    NUM_BUCKETS,
    ShapeConfig,
    bucket_index,
    build_config_tables,
    generate_config_header,
    generate_runtime_config,
    merge_runtime_configs,
)


# ---------------------------------------------------------------------------
# bucket_index tests
# ---------------------------------------------------------------------------


class TestBucketIndex:
    def test_below_first_boundary(self):
        assert bucket_index(1) == 0
        assert bucket_index(64) == 0
        assert bucket_index(128) == 0

    def test_exact_boundaries(self):
        for i, boundary in enumerate(BUCKET_BOUNDARIES):
            assert bucket_index(boundary) == i

    def test_just_above_boundaries(self):
        assert bucket_index(129) == 1
        assert bucket_index(1025) == 2
        assert bucket_index(4097) == 3
        assert bucket_index(16385) == 4

    def test_large_value(self):
        assert bucket_index(151936) == len(BUCKET_BOUNDARIES)

    def test_value_one(self):
        assert bucket_index(1) == 0

    def test_between_boundaries(self):
        assert bucket_index(500) == 1
        assert bucket_index(2048) == 2
        assert bucket_index(8192) == 3


# ---------------------------------------------------------------------------
# ShapeConfig tests
# ---------------------------------------------------------------------------


class TestShapeConfig:
    def test_frozen(self):
        cfg = ShapeConfig(nwarps=4, rows_per_block=2)
        with pytest.raises(AttributeError):
            cfg.nwarps = 8

    def test_fields(self):
        cfg = ShapeConfig(nwarps=8, rows_per_block=4)
        assert cfg.nwarps == 8
        assert cfg.rows_per_block == 4

    def test_equality(self):
        a = ShapeConfig(nwarps=4, rows_per_block=1)
        b = ShapeConfig(nwarps=4, rows_per_block=1)
        assert a == b

    def test_default_config(self):
        assert DEFAULT_CONFIG.nwarps == 4
        assert DEFAULT_CONFIG.rows_per_block == 1


# ---------------------------------------------------------------------------
# build_config_tables tests
# ---------------------------------------------------------------------------


class TestBuildConfigTables:
    def test_empty_configs(self):
        tables = build_config_tables({})
        assert tables == {}

    def test_single_entry(self):
        configs = {("Q4_K", 4096, 4096): {"nwarps": 8, "rows_per_block": 2}}
        tables = build_config_tables(configs)
        assert "Q4_K" in tables
        table = tables["Q4_K"]
        assert table[2][2] == ShapeConfig(nwarps=8, rows_per_block=2)

    def test_defaults_fill_empty_buckets(self):
        configs = {("Q4_K", 4096, 4096): {"nwarps": 8, "rows_per_block": 2}}
        tables = build_config_tables(configs)
        table = tables["Q4_K"]
        for ni in range(NUM_BUCKETS):
            for ki in range(NUM_BUCKETS):
                if (ni, ki) != (2, 2):
                    assert table[ni][ki] == DEFAULT_CONFIG

    def test_multiple_quant_types(self):
        configs = {
            ("Q4_K", 4096, 4096): {"nwarps": 4, "rows_per_block": 2},
            ("Q6_K", 4096, 4096): {"nwarps": 8, "rows_per_block": 4},
        }
        tables = build_config_tables(configs)
        assert "Q4_K" in tables
        assert "Q6_K" in tables
        assert tables["Q4_K"][2][2] == ShapeConfig(nwarps=4, rows_per_block=2)
        assert tables["Q6_K"][2][2] == ShapeConfig(nwarps=8, rows_per_block=4)

    def test_table_dimensions(self):
        configs = {("Q4_K", 64, 64): {"nwarps": 2, "rows_per_block": 1}}
        tables = build_config_tables(configs)
        table = tables["Q4_K"]
        assert len(table) == NUM_BUCKETS
        for row in table:
            assert len(row) == NUM_BUCKETS

    def test_large_dimensions_go_to_last_bucket(self):
        configs = {("Q4_K", 151936, 4096): {"nwarps": 8, "rows_per_block": 4}}
        tables = build_config_tables(configs)
        assert tables["Q4_K"][4][2] == ShapeConfig(nwarps=8, rows_per_block=4)

    def test_multiple_entries_same_bucket(self):
        configs = {
            ("Q4_K", 3000, 3000): {"nwarps": 2, "rows_per_block": 1},
            ("Q4_K", 4000, 4000): {"nwarps": 8, "rows_per_block": 4},
        }
        tables = build_config_tables(configs)
        cfg = tables["Q4_K"][2][2]
        assert cfg.nwarps in (2, 8)


# ---------------------------------------------------------------------------
# generate_config_header tests (GGML_TYPE_COUNT indexed format)
# ---------------------------------------------------------------------------


class TestGenerateConfigHeader:
    @pytest.fixture
    def sample_configs(self):
        return {
            ("Q4_K", 4096, 4096): {"nwarps": 4, "rows_per_block": 2},
            ("Q4_K", 4096, 1024): {"nwarps": 2, "rows_per_block": 1},
            ("Q4_K", 12288, 4096): {"nwarps": 8, "rows_per_block": 4},
            ("Q6_K", 4096, 1024): {"nwarps": 2, "rows_per_block": 1},
        }

    def test_header_contains_pragma_once(self, sample_configs):
        header = generate_config_header(sample_configs)
        assert "#pragma once" in header

    def test_header_contains_struct(self, sample_configs):
        header = generate_config_header(sample_configs)
        assert "struct smithy_shape_config {" in header
        assert "int nwarps;" in header
        assert "int rows_per_block;" in header

    def test_header_contains_gpu_name(self, sample_configs):
        header = generate_config_header(sample_configs, gpu_name="gfx1100 (7900 XTX)")
        assert "gfx1100 (7900 XTX)" in header

    def test_header_contains_model_name(self, sample_configs):
        header = generate_config_header(sample_configs, model_name="Qwen3-8B-Q4_K_M")
        assert "Qwen3-8B-Q4_K_M" in header

    def test_header_contains_bucket_enum(self, sample_configs):
        header = generate_config_header(sample_configs)
        assert "smithy_get_bucket" in header
        assert "SMITHY_NUM_BUCKETS" in header
        for boundary in BUCKET_BOUNDARIES:
            assert str(boundary) in header

    def test_header_has_ggml_type_count_array(self, sample_configs):
        header = generate_config_header(sample_configs)
        assert f"smithy_configs[{GGML_TYPE_COUNT}]" in header
        assert "SMITHY_CONFIG_TABLE" in header

    def test_header_has_type_comments(self, sample_configs):
        header = generate_config_header(sample_configs)
        assert "GGML_TYPE_Q4_K" in header
        assert "GGML_TYPE_Q6_K" in header

    def test_specific_config_values_in_output(self, sample_configs):
        header = generate_config_header(sample_configs)
        # Q4_K (4096, 1024) -> nwarps=2, rows_per_block=1 -> N bucket 2, K bucket 1
        assert "{ 2, 1}" in header
        # Q4_K (12288, 4096) -> nwarps=8, rows_per_block=4 -> N bucket 3, K bucket 2
        assert "{ 8, 4}" in header

    def test_empty_configs(self):
        header = generate_config_header({})
        assert "#pragma once" in header
        assert "struct smithy_shape_config {" in header

    def test_single_quant_type(self):
        configs = {("Q8_0", 4096, 4096): {"nwarps": 4, "rows_per_block": 2}}
        header = generate_config_header(configs)
        # Q8_0 is ggml_type 8, should have data at index [8]
        assert f"[{GGML_TYPE_MAP['Q8_0']}] = GGML_TYPE_Q8_0" in header

    def test_header_is_valid_looking_c(self, sample_configs):
        header = generate_config_header(sample_configs)
        assert header.count("{") == header.count("}")

    def test_unfilled_buckets_are_zero(self, sample_configs):
        header = generate_config_header(sample_configs)
        # Most entries should be {0, 0} (no override)
        assert header.count("{ 0, 0}") > 10

    def test_unused_types_are_empty(self, sample_configs):
        header = generate_config_header(sample_configs)
        assert "unused" in header

    def test_ggml_type_count_dimensions(self, sample_configs):
        header = generate_config_header(sample_configs)
        assert "[SMITHY_NUM_BUCKETS][SMITHY_NUM_BUCKETS]" in header


# ---------------------------------------------------------------------------
# Integration: GGUF -> codegen pipeline
# ---------------------------------------------------------------------------


class TestGGUFToCodegen:
    def test_gguf_shapes_to_configs(self):
        unique_shapes = {
            ("Q4_K", 4096, 1024): 54,
            ("Q4_K", 4096, 4096): 72,
            ("Q4_K", 4096, 12288): 72,
            ("Q4_K", 4096, 151936): 1,
            ("Q4_K", 12288, 4096): 18,
            ("Q6_K", 4096, 1024): 18,
            ("Q6_K", 4096, 151936): 1,
            ("Q6_K", 12288, 4096): 18,
        }

        configs = {}
        for (qt, n, k), _count in unique_shapes.items():
            configs[(qt, n, k)] = {"nwarps": 4, "rows_per_block": 1}

        header = generate_config_header(
            configs,
            gpu_name="gfx1100 (7900 XTX)",
            model_name="Qwen3-8B-Q4_K_M",
        )

        assert "Qwen3-8B-Q4_K_M" in header
        assert "GGML_TYPE_Q4_K" in header
        assert "GGML_TYPE_Q6_K" in header
        assert "#pragma once" in header
        assert f"smithy_configs[{GGML_TYPE_COUNT}]" in header


# ---------------------------------------------------------------------------
# Priority-aware merge (speculative decoding target+draft)
# ---------------------------------------------------------------------------


class TestBuildConfigTablesWithPriority:
    def test_no_priority_falls_back_to_last_seen_wins(self):
        # Two shapes that share a bucket cell -- (LE_4096, LE_4096) for both.
        configs = {
            ("Q4_K", 3000, 3000): {"nwarps": 2, "rows_per_block": 1},
            ("Q4_K", 4000, 4000): {"nwarps": 8, "rows_per_block": 4},
        }
        tables = build_config_tables(configs)
        # Last inserted wins when no priorities are given.
        assert tables["Q4_K"][2][2] == ShapeConfig(nwarps=8, rows_per_block=4)

    def test_priority_picks_higher_speedup(self):
        configs = {
            ("Q4_K", 3000, 3000): {"nwarps": 2, "rows_per_block": 1},
            ("Q4_K", 4000, 4000): {"nwarps": 8, "rows_per_block": 4},
        }
        priorities = {
            ("Q4_K", 3000, 3000): 1.95,  # better speedup -- should win
            ("Q4_K", 4000, 4000): 1.10,
        }
        tables = build_config_tables(configs, priorities=priorities)
        assert tables["Q4_K"][2][2] == ShapeConfig(nwarps=2, rows_per_block=1)

    def test_priority_missing_treated_as_minus_inf(self):
        # Only the first shape has a recorded speedup; second is "unknown" and
        # MUST NOT win on the bucket cell.
        configs = {
            ("Q4_K", 3000, 3000): {"nwarps": 2, "rows_per_block": 1},
            ("Q4_K", 4000, 4000): {"nwarps": 8, "rows_per_block": 4},
        }
        priorities = {("Q4_K", 3000, 3000): 1.30}
        tables = build_config_tables(configs, priorities=priorities)
        assert tables["Q4_K"][2][2] == ShapeConfig(nwarps=2, rows_per_block=1)

    def test_priority_does_not_affect_distinct_buckets(self):
        # Target-shaped (LE_16384) and draft-shaped (LE_1024) land in different
        # cells; both should survive regardless of priorities.
        configs = {
            ("Q4_K", 5120, 5120): {"nwarps": 8, "rows_per_block": 2},   # target
            ("Q4_K", 1024, 1024): {"nwarps": 2, "rows_per_block": 1},   # draft
        }
        priorities = {
            ("Q4_K", 5120, 5120): 1.80,
            ("Q4_K", 1024, 1024): 1.20,
        }
        tables = build_config_tables(configs, priorities=priorities)
        # 5120 -> bucket 3 (LE_16384), 1024 -> bucket 1 (LE_1024)
        assert tables["Q4_K"][3][3] == ShapeConfig(nwarps=8, rows_per_block=2)
        assert tables["Q4_K"][1][1] == ShapeConfig(nwarps=2, rows_per_block=1)

    def test_priority_threads_through_runtime_config(self):
        import json
        configs = {
            ("Q4_K", 3000, 3000): {"nwarps": 2, "rows_per_block": 1},
            ("Q4_K", 4000, 4000): {"nwarps": 8, "rows_per_block": 4},
        }
        priorities = {
            ("Q4_K", 3000, 3000): 1.95,
            ("Q4_K", 4000, 4000): 1.10,
        }
        out = json.loads(generate_runtime_config(configs, priorities=priorities))
        type_idx = str(GGML_TYPE_MAP["Q4_K"])
        assert out["configs"][type_idx]["2,2"] == {"nwarps": 2, "rows_per_block": 1}


# ---------------------------------------------------------------------------
# merge_runtime_configs (speculative decoding via separate JSONs)
# ---------------------------------------------------------------------------


class TestMergeRuntimeConfigs:
    def _payload(self, model, cells_by_type):
        return {
            "gpu": "gfx1100",
            "model": model,
            "configs": {
                str(t): {k: dict(v) for k, v in cells.items()}
                for t, cells in cells_by_type.items()
            },
        }

    def test_empty_inputs_returns_empty_payload(self):
        merged = merge_runtime_configs([])
        assert merged["configs"] == {}

    def test_disjoint_cells_are_unioned(self):
        a = self._payload("target", {12: {"3,3": {"nwarps": 8, "rows_per_block": 2}}})
        b = self._payload("draft",  {12: {"1,1": {"nwarps": 2, "rows_per_block": 1}}})
        merged = merge_runtime_configs([a, b])
        cells = merged["configs"]["12"]
        assert cells["3,3"] == {"nwarps": 8, "rows_per_block": 2}
        assert cells["1,1"] == {"nwarps": 2, "rows_per_block": 1}

    def test_first_seen_wins_on_conflict(self):
        a = self._payload("target", {12: {"2,2": {"nwarps": 8, "rows_per_block": 4}}})
        b = self._payload("draft",  {12: {"2,2": {"nwarps": 2, "rows_per_block": 1}}})
        merged = merge_runtime_configs([a, b])
        # First payload (target) wins.
        assert merged["configs"]["12"]["2,2"] == {"nwarps": 8, "rows_per_block": 4}

    def test_distinct_quant_types_coexist(self):
        a = self._payload("target", {12: {"2,2": {"nwarps": 4, "rows_per_block": 1}}})
        b = self._payload("draft",  {14: {"1,1": {"nwarps": 2, "rows_per_block": 2}}})
        merged = merge_runtime_configs([a, b])
        assert "12" in merged["configs"]
        assert "14" in merged["configs"]

    def test_default_model_name_joins_with_plus(self):
        a = self._payload("Qwen3-8B", {12: {"2,2": {"nwarps": 4, "rows_per_block": 1}}})
        b = self._payload("Qwen3-0.6B", {12: {"1,1": {"nwarps": 2, "rows_per_block": 1}}})
        merged = merge_runtime_configs([a, b])
        assert merged["model"] == "Qwen3-8B+Qwen3-0.6B"

    def test_explicit_overrides(self):
        a = self._payload("target", {12: {"2,2": {"nwarps": 4, "rows_per_block": 1}}})
        merged = merge_runtime_configs([a], gpu_name="gfx1201", model_name="custom")
        assert merged["gpu"] == "gfx1201"
        assert merged["model"] == "custom"

    def test_does_not_mutate_inputs(self):
        a = self._payload("target", {12: {"2,2": {"nwarps": 8, "rows_per_block": 4}}})
        b = self._payload("draft",  {12: {"2,2": {"nwarps": 2, "rows_per_block": 1}}})
        a_snapshot = {"gpu": a["gpu"], "model": a["model"], "configs": {k: dict(v) for k, v in a["configs"].items()}}
        merge_runtime_configs([a, b])
        assert a["configs"]["12"]["2,2"] == a_snapshot["configs"]["12"]["2,2"]


# ---------------------------------------------------------------------------
# CLI: merge-configs subcommand end-to-end
# ---------------------------------------------------------------------------


class TestCLIMergeConfigs:
    def test_merge_configs_writes_output(self, tmp_path, monkeypatch):
        import json
        from kernel_anvil import cli

        a = tmp_path / "target.json"
        b = tmp_path / "draft.json"
        a.write_text(json.dumps({
            "gpu": "gfx1100", "model": "target",
            "configs": {"12": {"3,3": {"nwarps": 8, "rows_per_block": 2}}},
        }))
        b.write_text(json.dumps({
            "gpu": "gfx1100", "model": "draft",
            "configs": {"12": {"1,1": {"nwarps": 2, "rows_per_block": 1}}},
        }))
        out = tmp_path / "merged.json"

        monkeypatch.setattr(sys, "argv", [
            "kernel-anvil", "merge-configs", str(a), str(b), "-o", str(out),
        ])
        cli.main()

        assert out.exists()
        merged = json.loads(out.read_text())
        cells = merged["configs"]["12"]
        assert cells["3,3"] == {"nwarps": 8, "rows_per_block": 2}
        assert cells["1,1"] == {"nwarps": 2, "rows_per_block": 1}
        assert merged["model"] == "target+draft"

    def test_merge_configs_missing_input_exits_clean(self, tmp_path, monkeypatch, capsys):
        from kernel_anvil import cli

        out = tmp_path / "merged.json"
        monkeypatch.setattr(sys, "argv", [
            "kernel-anvil", "merge-configs",
            str(tmp_path / "does-not-exist.json"),
            "-o", str(out),
        ])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1

    def test_merge_configs_invalid_json_exits_clean(self, tmp_path, monkeypatch):
        from kernel_anvil import cli

        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        out = tmp_path / "merged.json"
        monkeypatch.setattr(sys, "argv", [
            "kernel-anvil", "merge-configs", str(bad), "-o", str(out),
        ])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        assert not out.exists()

    def test_merge_configs_atomic_write(self, tmp_path, monkeypatch):
        # No partial / temp files left in the parent dir after a successful merge.
        import json
        from kernel_anvil import cli

        a = tmp_path / "a.json"
        a.write_text(json.dumps({
            "gpu": "gfx1100", "model": "x",
            "configs": {"12": {"2,2": {"nwarps": 4, "rows_per_block": 1}}},
        }))
        out = tmp_path / "merged.json"
        monkeypatch.setattr(sys, "argv", [
            "kernel-anvil", "merge-configs", str(a), "-o", str(out),
        ])
        cli.main()
        leftover = [p.name for p in tmp_path.iterdir() if p.name.endswith(".json.tmp")]
        assert leftover == [], f"atomic write left tempfiles behind: {leftover}"


# ---------------------------------------------------------------------------
# Defensive merge_runtime_configs: malformed payloads must not crash
# ---------------------------------------------------------------------------


class TestMergeRuntimeConfigsDefensive:
    def test_skips_non_dict_payload(self):
        merged = merge_runtime_configs([None, "not a dict", 42])
        assert merged["configs"] == {}

    def test_skips_non_dict_configs_section(self):
        bad = {"gpu": "x", "model": "y", "configs": "not a dict"}
        merged = merge_runtime_configs([bad])
        assert merged["configs"] == {}

    def test_skips_null_cells_section(self):
        bad = {"gpu": "x", "model": "y", "configs": {"12": None}}
        merged = merge_runtime_configs([bad])
        # Type bucket may be created empty or skipped; neither should crash.
        assert merged["configs"].get("12", {}) == {}

    def test_skips_string_cells_section(self):
        bad = {"gpu": "x", "model": "y", "configs": {"12": "evil"}}
        merged = merge_runtime_configs([bad])
        assert merged["configs"].get("12", {}) == {}

    def test_skips_non_dict_cell_cfg(self):
        bad = {"gpu": "x", "model": "y", "configs": {"12": {"1,1": "bad"}}}
        merged = merge_runtime_configs([bad])
        # Cell skipped; type bucket exists but is empty.
        assert merged["configs"].get("12", {}) == {}

    def test_null_model_field_does_not_crash_join(self):
        a = {"gpu": "x", "model": None, "configs": {}}
        b = {"gpu": "x", "model": "real", "configs": {}}
        merged = merge_runtime_configs([a, b])
        # None coerces to 'unknown', then joined.
        assert "+" in merged["model"]

    def test_well_formed_payload_after_malformed_still_merges(self):
        good = {"gpu": "x", "model": "good",
                "configs": {"12": {"2,2": {"nwarps": 4, "rows_per_block": 1}}}}
        merged = merge_runtime_configs([None, "junk", good])
        assert merged["configs"]["12"]["2,2"] == {"nwarps": 4, "rows_per_block": 1}


# ---------------------------------------------------------------------------
# NaN priority handling in build_config_tables
# ---------------------------------------------------------------------------


class TestNaNPriority:
    def test_nan_priority_does_not_silently_overwrite(self):
        # Without the NaN guard, a NaN priority always "wins" because
        # real_val <= nan is False, letting later entries overwrite.
        configs = {
            ("Q4_K", 3000, 3000): {"nwarps": 2, "rows_per_block": 1},
            ("Q4_K", 4000, 4000): {"nwarps": 8, "rows_per_block": 4},
        }
        priorities = {
            ("Q4_K", 3000, 3000): float("nan"),  # malformed
            ("Q4_K", 4000, 4000): 1.10,           # real
        }
        tables = build_config_tables(configs, priorities=priorities)
        # Real priority must win over NaN in the shared bucket.
        assert tables["Q4_K"][2][2] == ShapeConfig(nwarps=8, rows_per_block=4)
