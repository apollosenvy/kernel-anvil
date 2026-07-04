"""Tests for cell_ablation: trigger math, candidate cells, decoy baseline,
config emission, and the ablation loop with an injected bench."""

import json

import pytest

from kernel_anvil.cell_ablation import (
    DECOY_QUANT,
    AblationReport,
    BenchResult,
    CandidateCell,
    candidate_cells,
    run_ablation,
    small_k_nwarps,
    small_k_trigger_fires,
    write_config,
)
from kernel_anvil.codegen import GGML_TYPE_MAP


class TestTriggerMath:
    """Hand-computed mirrors of mmvq.cu's should_use_small_k arithmetic."""

    def test_q4_k_small_k_fires(self):
        # Q4_K: qk=256 qi=32 vdr=2, nwarps=8 (rdna3_0 whitelist), warp 32.
        # blocks_per_iter_1warp = 2*32/32 = 2; threshold = 8*2 = 16 blocks
        # => fires for K < 16*256 = 4096.
        assert small_k_trigger_fires("Q4_K", 32)
        assert small_k_trigger_fires("Q4_K", 512)
        assert small_k_trigger_fires("Q4_K", 2048)
        assert not small_k_trigger_fires("Q4_K", 4096)
        assert not small_k_trigger_fires("Q4_K", 8192)

    def test_q6_k_narrower_window(self):
        # Q6_K: vdr=1 -> bpi=1; threshold = 8 blocks => K < 2048.
        assert small_k_trigger_fires("Q6_K", 512)
        assert not small_k_trigger_fires("Q6_K", 2048)
        assert not small_k_trigger_fires("Q6_K", 8192)

    def test_non_whitelisted_type_never_fires(self):
        # Q2_K has nwarps=1 on rdna3_0 -> gate fails regardless of K.
        assert small_k_nwarps("Q2_K", "rdna3_0") == 1
        assert not small_k_trigger_fires("Q2_K", 32, "rdna3_0")
        # ...but it IS whitelisted on rdna4.
        assert small_k_trigger_fires("Q2_K", 32, "rdna4")

    def test_unknown_quant_never_fires(self):
        assert not small_k_trigger_fires("F32", 32)
        assert not small_k_trigger_fires("TURBO3_0", 32)


class TestCandidateCells:
    def test_bucket_dedupe_and_shape_capture(self):
        shapes = {
            ("Q4_K", 2048, 32): 60,
            ("Q4_K", 2048, 512): 94,   # different K bucket than 32
            ("Q4_K", 2048, 8192): 26,  # does not fire
            ("Q6_K", 2048, 512): 6,
            ("F32", 4, 8192): 30,      # unknown to trigger, skipped
        }
        cells = candidate_cells(shapes)
        keys = {c.key for c in cells}
        assert keys == {"Q4_K:2,0", "Q4_K:2,1", "Q6_K:2,1"}
        by_key = {c.key: c for c in cells}
        assert by_key["Q4_K:2,1"].shapes == [(2048, 512, 94)]

    def test_same_bucket_shapes_merge(self):
        shapes = {
            ("Q4_K", 2048, 300): 10,
            ("Q4_K", 2048, 512): 20,  # both K in (128, 1024] bucket
        }
        cells = candidate_cells(shapes)
        assert len(cells) == 1
        assert len(cells[0].shapes) == 2


class TestConfigEmission:
    def test_empty_cells_writes_decoy_not_empty(self, tmp_path):
        """An empty config would fall through the runtime loader chain to
        default.json and contaminate the baseline; the decoy prevents it."""
        out = tmp_path / "baseline.json"
        write_config(out, [], "rdna3_0")
        data = json.loads(out.read_text())
        assert data["configs"], "config must never be empty"
        decoy_idx = str(GGML_TYPE_MAP[DECOY_QUANT])
        assert list(data["configs"].keys()) == [decoy_idx]

    def test_winner_cell_uses_real_nwarps(self, tmp_path):
        out = tmp_path / "cfg.json"
        cell = CandidateCell("Q4_K", 2, 1, shapes=[(2048, 512, 94)])
        write_config(out, [cell], "rdna3_0")
        data = json.loads(out.read_text())
        entry = data["configs"][str(GGML_TYPE_MAP["Q4_K"])]["2,1"]
        # rpb mirrors the runtime's actual small_k rows (nwarps), not a
        # proxy-derived BLOCK_N//64.
        assert entry == {"nwarps": 8, "rows_per_block": 8}


class TestAblationLoop:
    SHAPES = {
        ("Q4_K", 2048, 32): 60,
        ("Q4_K", 2048, 512): 94,
        ("Q6_K", 2048, 512): 6,
    }

    def test_keeps_only_cells_above_threshold(self, tmp_path):
        # Baseline 100 tok/s; one cell wins (+5%), one flat, one regresses.
        speeds = iter([100.0, 105.0, 100.2, 97.0])

        def fake_bench(cfg_path):
            return BenchResult(tokens_per_second=next(speeds), raw={})

        out = tmp_path / "final.json"
        report = run_ablation(
            "model.gguf", "llama-bench", self.SHAPES,
            threshold=1.02, out_path=out, bench_fn=fake_bench, log=lambda *_: None)
        assert isinstance(report, AblationReport)
        assert report.baseline_ts == 100.0
        assert len(report.cell_results) == 3
        assert len(report.winners) == 1
        data = json.loads(out.read_text())
        # Exactly one real cell in the final config.
        cells = [c for t in data["configs"].values() for c in t]
        assert len(cells) == 1

    def test_no_winners_emits_decoy_config(self, tmp_path):
        def fake_bench(cfg_path):
            return BenchResult(tokens_per_second=100.0, raw={})

        out = tmp_path / "final.json"
        report = run_ablation(
            "model.gguf", "llama-bench", self.SHAPES,
            threshold=1.02, out_path=out, bench_fn=fake_bench, log=lambda *_: None)
        assert report.winners == []
        data = json.loads(out.read_text())
        assert str(GGML_TYPE_MAP[DECOY_QUANT]) in data["configs"]

    def test_no_candidates_short_circuits(self):
        report = run_ablation(
            "model.gguf", "llama-bench", {("Q4_K", 2048, 8192): 26},
            bench_fn=lambda _: pytest.fail("bench must not run"),
            log=lambda *_: None)
        assert report.cell_results == []
        assert report.config_path is None
