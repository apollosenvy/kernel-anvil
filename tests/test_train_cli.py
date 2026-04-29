"""Tests for the `kernel-anvil train-optimize` CLI subcommand.

These tests exercise the JSON pipeline end-to-end without requiring a GPU
or transformers. The HF shape extractor is monkey-patched to a stub, and
``--dry-run`` skips the runner sweep entirely.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from kernel_anvil import cli


PROJ_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# argparse plumbing
# ---------------------------------------------------------------------------


class TestTrainOptimizeArgs:
    def test_help_via_subprocess(self):
        result = subprocess.run(
            [sys.executable, "-m", "kernel_anvil.cli", "train-optimize", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "train-optimize" in result.stdout
        assert "--quant" in result.stdout
        assert "--batch" in result.stdout
        assert "--seq" in result.stdout
        assert "--dry-run" in result.stdout

    def test_missing_model_arg_fails(self):
        result = subprocess.run(
            [sys.executable, "-m", "kernel_anvil.cli", "train-optimize",
             "--quant", "mxfp4", "--batch", "1", "--seq", "4096", "--dry-run"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0

    def test_missing_quant_fails(self):
        result = subprocess.run(
            [sys.executable, "-m", "kernel_anvil.cli", "train-optimize",
             "Qwen3-8B", "--batch", "1", "--seq", "4096", "--dry-run"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# end-to-end dry-run
# ---------------------------------------------------------------------------


def _stub_shapes(model_id_or_config, *, batch: int, seq: int, ops=None):
    """Replacement for extract_shapes -- returns deterministic Qwen3-8B-ish set."""
    M = batch * seq
    base = [
        (M, 4096, 4096),
        (M, 12288, 4096),
        (M, 4096, 12288),
        (M, 1024, 4096),
    ]
    op_list = list(ops) if ops else ["mxfp4_fwd", "mxfp4_grad_input"]
    out = []
    for op in op_list:
        for m, n, k in base:
            out.append((op, m, n, k))
    return out


class TestTrainOptimizeDryRun:
    def test_writes_json_to_default_path(self, tmp_path, monkeypatch):
        # Force a deterministic cache root by overriding HOME
        monkeypatch.setenv("HOME", str(tmp_path))
        # Stub shape extraction so we don't need transformers
        monkeypatch.setattr("kernel_anvil.train_shapes.extract_shapes", _stub_shapes)
        # Run the CLI
        monkeypatch.setattr(sys, "argv", [
            "kernel-anvil", "train-optimize", "Qwen3-8B",
            "--quant", "mxfp4", "--batch", "1", "--seq", "4096", "--dry-run",
        ])
        cli.main()
        # Cache path should be ~/.cache/anvil-train/<gpu>/Qwen3-8B-mxfp4-b1s4096.json
        cache_root = tmp_path / ".cache" / "anvil-train"
        assert cache_root.exists()
        # Find the GPU subdir (e.g. gfx1100)
        gpu_dirs = list(cache_root.iterdir())
        assert gpu_dirs, "expected a GPU subdir under ~/.cache/anvil-train"
        json_files = list(gpu_dirs[0].glob("*.json"))
        assert json_files, "expected a JSON file in the GPU subdir"
        payload = json.loads(json_files[0].read_text())
        assert payload["schema"] == "anvil-train/v1"
        assert payload["model"] == "Qwen3-8B"
        assert payload["batch"] == 1
        assert payload["seq"] == 4096
        # Dry-run uses placeholder ops
        assert "mxfp4_fwd" in payload["ops"]
        assert "mxfp4_grad_input" in payload["ops"]

    def test_writes_json_to_explicit_output(self, tmp_path, monkeypatch):
        out_path = tmp_path / "custom.json"
        monkeypatch.setattr("kernel_anvil.train_shapes.extract_shapes", _stub_shapes)
        monkeypatch.setattr(sys, "argv", [
            "kernel-anvil", "train-optimize", "Qwen3-8B",
            "--quant", "mxfp4", "--batch", "1", "--seq", "4096",
            "--dry-run", "--output", str(out_path),
        ])
        cli.main()
        assert out_path.exists()
        payload = json.loads(out_path.read_text())
        assert payload["schema"] == "anvil-train/v1"

    def test_dry_run_payload_has_placeholder_configs(self, tmp_path, monkeypatch):
        out_path = tmp_path / "out.json"
        monkeypatch.setattr("kernel_anvil.train_shapes.extract_shapes", _stub_shapes)
        monkeypatch.setattr(sys, "argv", [
            "kernel-anvil", "train-optimize", "Qwen3-8B",
            "--quant", "mxfp4", "--batch", "1", "--seq", "4096",
            "--dry-run", "--output", str(out_path),
        ])
        cli.main()
        payload = json.loads(out_path.read_text())
        # Every cell must have the six required keys.
        for op, cells in payload["ops"].items():
            for cell, body in cells.items():
                for k in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M",
                          "num_warps", "num_stages"):
                    assert k in body, f"missing {k} in {op}/{cell}"

    def test_dry_run_int4_op_set(self, tmp_path, monkeypatch):
        out_path = tmp_path / "int4.json"
        monkeypatch.setattr("kernel_anvil.train_shapes.extract_shapes", _stub_shapes)
        monkeypatch.setattr(sys, "argv", [
            "kernel-anvil", "train-optimize", "Qwen3-8B",
            "--quant", "int4", "--batch", "1", "--seq", "4096",
            "--dry-run", "--output", str(out_path),
        ])
        cli.main()
        payload = json.loads(out_path.read_text())
        # int4 quant -> default ops should be int4_fwd + int4_grad_input
        assert any(op.startswith("int4") for op in payload["ops"])

    def test_extract_shapes_failure_exits_clean(self, tmp_path, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("network down")

        monkeypatch.setattr("kernel_anvil.train_shapes.extract_shapes", boom)
        out_path = tmp_path / "out.json"
        monkeypatch.setattr(sys, "argv", [
            "kernel-anvil", "train-optimize", "Qwen3-8B",
            "--quant", "mxfp4", "--batch", "1", "--seq", "4096",
            "--dry-run", "--output", str(out_path),
        ])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1

    def test_atomic_write_no_tmp_left(self, tmp_path, monkeypatch):
        out_path = tmp_path / "out.json"
        monkeypatch.setattr("kernel_anvil.train_shapes.extract_shapes", _stub_shapes)
        monkeypatch.setattr(sys, "argv", [
            "kernel-anvil", "train-optimize", "Qwen3-8B",
            "--quant", "mxfp4", "--batch", "1", "--seq", "4096",
            "--dry-run", "--output", str(out_path),
        ])
        cli.main()
        leftover = [p.name for p in tmp_path.iterdir() if p.name.endswith(".json.tmp")]
        assert leftover == [], f"atomic write left tempfiles: {leftover}"


# ---------------------------------------------------------------------------
# train-optimize parameter space sanity (smoke test)
# ---------------------------------------------------------------------------


class TestTrainParamSpace:
    def test_generates_at_most_max_configs(self):
        from kernel_anvil.train_param_space import generate_train_configs

        configs = generate_train_configs(max_configs=10)
        assert len(configs) <= 10
        assert all(isinstance(c, dict) for c in configs)
        for c in configs:
            for k in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M",
                      "num_warps", "num_stages"):
                assert k in c

    def test_default_returns_about_30(self):
        from kernel_anvil.train_param_space import generate_train_configs

        configs = generate_train_configs()
        # Cap is 30 by default; the filter MAY trim further but the cap is
        # the upper bound.
        assert 1 <= len(configs) <= 30

    def test_rdna3_filter_drops_oversize_configs(self):
        from kernel_anvil.train_param_space import generate_train_configs
        from kernel_anvil.rdna3 import GFX1100

        # Force a tiny base grid that will all overflow LDS budget.
        configs = generate_train_configs(
            gpu=GFX1100,
            block_m_values=[256],
            block_n_values=[256],
            block_k_values=[128],
            group_m_values=[8],
            num_warps_values=[8],
            num_stages_values=[4],
        )
        # 256 * 128 * 2 + 256 * 128 * 1 = 65536 + 32768 = 98304 *4 stages
        # >> 98304 LDS budget, so should be filtered.
        assert configs == []
