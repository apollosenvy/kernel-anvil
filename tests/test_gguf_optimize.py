"""Tests for the smithy gguf-optimize CLI command."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from kernel_anvil.cli import main, _make_runner, _tune_shape_cli

PROJ_ROOT = Path(__file__).parent.parent
# Override with KERNEL_ANVIL_TEST_GGUF=/path/to/any.gguf to point the
# integration tests at a different model. Tests skip cleanly when absent.
QWEN3_PATH = Path(
    os.environ.get(
        "KERNEL_ANVIL_TEST_GGUF",
        str(Path.home() / "Models" / "Qwen3-8B" / "Qwen3-8B-Q4_K_M.gguf"),
    )
)
HAS_GPU = torch.cuda.is_available()

_skip_no_gpu = pytest.mark.skipif(not HAS_GPU, reason="No GPU available")
_skip_no_model = pytest.mark.skipif(
    not QWEN3_PATH.exists(), reason=f"GGUF not found: {QWEN3_PATH}"
)


# ---------------------------------------------------------------------------
# CLI argument parsing and help
# ---------------------------------------------------------------------------


class TestGGUFOptimizeHelp:
    def test_help_flag(self):
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["kernel-anvil", "gguf-optimize", "--help"]):
                main()
        assert exc_info.value.code == 0

    def test_main_help_lists_gguf_optimize(self):
        result = subprocess.run(
            [sys.executable, "-m", "kernel_anvil.cli", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "gguf-optimize" in result.stdout

    def test_missing_gguf_arg(self):
        result = subprocess.run(
            [sys.executable, "-m", "kernel_anvil.cli", "gguf-optimize"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0

    def test_nonexistent_gguf_file(self):
        result = subprocess.run(
            [sys.executable, "-m", "kernel_anvil.cli", "gguf-optimize", "/nonexistent/model.gguf"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# _make_runner unit tests (require GPU)
# ---------------------------------------------------------------------------


@_skip_no_gpu
class TestMakeRunner:
    def test_returns_callable_ref_and_bytes(self):
        kernel_fn, ref, data_bytes = _make_runner(64, 128, torch.device("cuda"))
        assert callable(kernel_fn)
        assert isinstance(ref, torch.Tensor)
        assert ref.shape == (64,)
        assert data_bytes == (64 * 128 + 128 + 64) * 2

    def test_kernel_fn_runs(self):
        kernel_fn, ref, _ = _make_runner(64, 128, torch.device("cuda"))
        config = {"BLOCK_N": 64, "BLOCK_K": 128, "num_warps": 4, "num_stages": 1}
        out = kernel_fn(config)
        assert out.shape == ref.shape
        assert out.device == ref.device

    def test_kernel_fn_output_reasonable(self):
        kernel_fn, ref, _ = _make_runner(64, 128, torch.device("cuda"))
        config = {"BLOCK_N": 64, "BLOCK_K": 128, "num_warps": 4, "num_stages": 1}
        out = kernel_fn(config)
        assert torch.allclose(out, ref, atol=1e-1, rtol=1e-1)


# ---------------------------------------------------------------------------
# _tune_shape_cli unit tests (require GPU)
# ---------------------------------------------------------------------------


@_skip_no_gpu
class TestTuneShapeCli:
    def test_returns_config_and_metrics(self):
        cfg, baseline_us, speedup = _tune_shape_cli(
            N=64, K=128,
            device=torch.device("cuda"),
            gpu_spec=None,
            max_configs=3,
            warmup=1,
            runs=2,
        )
        assert "nwarps" in cfg
        assert "rows_per_block" in cfg
        assert isinstance(cfg["nwarps"], int)
        assert isinstance(cfg["rows_per_block"], int)
        assert cfg["nwarps"] > 0
        assert cfg["rows_per_block"] >= 1
        assert baseline_us > 0

    def test_speedup_at_least_one(self):
        cfg, _, speedup = _tune_shape_cli(
            N=128, K=256,
            device=torch.device("cuda"),
            gpu_spec=None,
            max_configs=3,
            warmup=1,
            runs=2,
        )
        # Speedup should be >= ~0.5 (best should be no worse than 2x slower than baseline)
        assert speedup is not None
        assert speedup > 0.5


# ---------------------------------------------------------------------------
# End-to-end tests (require GPU + model)
# ---------------------------------------------------------------------------


@_skip_no_gpu
@_skip_no_model
class TestGGUFOptimizeEndToEnd:
    """Full end-to-end test with the real Qwen3 GGUF.

    This is slow (~minutes) since it tunes every unique shape.
    Only runs when GPU and model are both available.
    """

    def test_generates_header(self, tmp_path):
        output = tmp_path / "smithy-config.h"
        result = subprocess.run(
            [
                sys.executable, "-m", "kernel_anvil.cli", "gguf-optimize",
                str(QWEN3_PATH),
                "--output", str(output),
                "--max-configs", "3",
                "--warmup", "1",
                "--runs", "2",
            ],
            capture_output=True, text=True, timeout=600,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output.exists()
        header = output.read_text()
        assert "#pragma once" in header
        assert "smithy_shape_config" in header
        assert "Qwen3" in header


# ---------------------------------------------------------------------------
# Tests with mocked GPU (no GPU required)
# ---------------------------------------------------------------------------


class TestGGUFOptimizeMocked:
    """Test the command flow with mocked tuning (no GPU needed)."""

    def test_no_gpu_falls_back_to_no_bench(self, tmp_path, monkeypatch, capsys):
        """When torch.cuda.is_available() returns False, gguf-optimize must
        fall back to --no-bench mode (heuristic configs) instead of failing.
        Prior to this rewrite, this test only checked --help and didn't
        actually exercise the no-GPU code path at cli.py:422-428."""
        from kernel_anvil.gguf import ModelProfile, TensorInfo
        from kernel_anvil.cli import cmd_gguf_optimize
        import argparse

        monkeypatch.setenv("HOME", str(tmp_path))  # sandbox cache dir
        fake_gguf = tmp_path / "fake.gguf"
        fake_gguf.write_bytes(b"GGUF" + b"\x00" * 100)
        profile = ModelProfile(
            name="NoGpuTest",
            architecture="test",
            tensors=[TensorInfo("w", (64, 64), "Q4_K", 4096)],
            unique_shapes={("Q4_K", 64, 64): 1},
        )
        args = argparse.Namespace(
            gguf=str(fake_gguf), output="smithy-config.h",
            max_configs=3, warmup=1, runs=2,
            no_bench=False, draft=None,
        )

        with patch("kernel_anvil.cli.parse_gguf", return_value=profile), \
             patch("torch.cuda.is_available", return_value=False):
            cmd_gguf_optimize(args)

        out = capsys.readouterr().out
        assert "GPU not detected" in out  # the fallback branch was taken
        # And the cache file was still written (heuristic configs).
        cache = tmp_path / ".cache" / "smithy" / "fake.json"
        assert cache.exists()

    def test_nonexistent_file_exits(self):
        result = subprocess.run(
            [sys.executable, "-m", "kernel_anvil.cli", "gguf-optimize", "/tmp/nonexistent_12345.gguf"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Small-shape GPU test (fast, doesn't need real GGUF)
# ---------------------------------------------------------------------------


@_skip_no_gpu
class TestGGUFOptimizeSmallShape:
    """Test the pipeline with a synthetic small model profile."""

    @pytest.fixture(autouse=True)
    def _sandbox_home(self, tmp_path, monkeypatch):
        """Redirect ~/.cache/smithy/ writes to tmp_path so this test does
        not clobber the user's real config cache."""
        monkeypatch.setenv("HOME", str(tmp_path))

    def test_pipeline_with_mock_gguf(self, tmp_path):
        """Mock parse_gguf to return small shapes, then run the real pipeline."""
        from kernel_anvil.gguf import ModelProfile, TensorInfo

        small_profile = ModelProfile(
            name="TestModel-Tiny",
            architecture="test",
            tensors=[
                TensorInfo("w1", (64, 128), "Q4_K", 8192),
                TensorInfo("w2", (128, 64), "Q4_K", 8192),
            ],
            unique_shapes={
                ("Q4_K", 64, 128): 1,
                ("Q4_K", 128, 64): 1,
            },
        )

        output = tmp_path / "test-config.h"
        # Create a fake GGUF file so the path check passes
        fake_gguf = tmp_path / "test.gguf"
        fake_gguf.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("kernel_anvil.cli.parse_gguf", return_value=small_profile):
            from kernel_anvil.cli import cmd_gguf_optimize
            import argparse
            # Match what argparse actually produces: include all flags so
            # this test fails loudly if direct attribute access ever
            # replaces the getattr() defensive defaults in cmd_gguf_optimize.
            args = argparse.Namespace(
                gguf=str(fake_gguf),
                output=str(output),
                max_configs=3,
                warmup=1,
                runs=2,
                no_bench=False,
                draft=None,
            )

            cmd_gguf_optimize(args)

        assert output.exists()
        header = output.read_text()
        assert "#pragma once" in header
        assert "GGML_TYPE_Q4_K" in header
        assert "TestModel-Tiny" in header


# ---------------------------------------------------------------------------
# --draft (speculative decoding) flow
# ---------------------------------------------------------------------------


class TestGGUFOptimizeDraft:
    def _make_profile(self, name, shapes):
        from kernel_anvil.gguf import ModelProfile, TensorInfo
        return ModelProfile(
            name=name,
            architecture="test",
            tensors=[TensorInfo(f"w{i}", (n, k), qt, n * k)
                     for i, (qt, n, k) in enumerate(shapes)],
            unique_shapes={s: 1 for s in shapes},
        )

    @pytest.fixture(autouse=True)
    def _sandbox_home(self, tmp_path, monkeypatch):
        """Redirect ~/.cache/smithy/ writes to tmp_path so tests don't
        clobber the user's real cache. cmd_gguf_optimize uses
        Path.home() which honors $HOME on POSIX."""
        monkeypatch.setenv("HOME", str(tmp_path))

    def _run(self, tmp_path, target_shapes, draft_shapes_list):
        """Invoke cmd_gguf_optimize with mocked GGUFs and --no-bench."""
        import argparse

        target_gguf = tmp_path / "target.gguf"
        target_gguf.write_bytes(b"GGUF" + b"\x00" * 100)
        target_profile = self._make_profile("Target-Model", target_shapes)

        draft_paths = []
        draft_profiles = []
        for i, ds in enumerate(draft_shapes_list, start=1):
            dp = tmp_path / f"draft{i}.gguf"
            dp.write_bytes(b"GGUF" + b"\x00" * 100)
            draft_paths.append(str(dp))
            draft_profiles.append(self._make_profile(f"Draft-{i}", ds))

        # parse_gguf returns target first, then each draft in order.
        parse_returns = [target_profile, *draft_profiles]

        # Force --no-bench so we don't need a GPU.
        args = argparse.Namespace(
            gguf=str(target_gguf),
            output="smithy-config.h",
            max_configs=3,
            warmup=1,
            runs=2,
            no_bench=True,
            draft=draft_paths or None,
        )

        from kernel_anvil.cli import cmd_gguf_optimize
        with patch("kernel_anvil.cli.parse_gguf", side_effect=parse_returns):
            cmd_gguf_optimize(args)

        # Reads from sandboxed HOME (set by _sandbox_home fixture).
        cache_path = Path.home() / ".cache" / "smithy" / f"{Path(target_gguf).stem}.json"
        return cache_path

    def test_draft_writes_merged_model_name(self, tmp_path, capsys):
        cache = self._run(
            tmp_path,
            target_shapes=[("Q4_K", 4096, 4096)],
            draft_shapes_list=[[("Q4_K", 1024, 1024)]],
        )
        import json
        data = json.loads(cache.read_text())
        assert data["model"] == "Target-Model+Draft-1"
        # Sandboxed cache must be inside tmp_path -- never at the real ~.
        assert str(tmp_path) in str(cache)

    def test_draft_run_hint_includes_md_flag(self, tmp_path, capsys):
        # The "Run:" line printed at the end MUST include -md <draft_path>
        # when --draft is used, otherwise users copy-pasting the hint will
        # silently disable speculative decoding.
        self._run(
            tmp_path,
            target_shapes=[("Q4_K", 4096, 4096)],
            draft_shapes_list=[[("Q4_K", 1024, 1024)]],
        )
        out = capsys.readouterr().out
        assert "-md" in out
        assert "draft1.gguf" in out
        # Should also use SMITHY_CONFIG (explicit path) rather than
        # SMITHY_MODEL when --draft is used.
        assert "SMITHY_CONFIG=" in out

    def test_no_draft_hint_omits_md(self, tmp_path, capsys):
        # Without --draft, the hint must NOT contain -md (regression check).
        self._run(
            tmp_path,
            target_shapes=[("Q4_K", 4096, 4096)],
            draft_shapes_list=[],
        )
        out = capsys.readouterr().out
        assert "-md" not in out

    def test_multiple_drafts_produce_labeled_results(self, tmp_path):
        cache = self._run(
            tmp_path,
            target_shapes=[("Q4_K", 4096, 4096)],
            draft_shapes_list=[
                [("Q4_K", 1024, 1024)],
                [("Q4_K", 2048, 2048)],
            ],
        )
        import json
        data = json.loads(cache.read_text())
        assert "Draft-1" in data["model"]
        assert "Draft-2" in data["model"]

    def test_draft_recovers_after_target_profile_failure(self, tmp_path):
        # If profiling raises for shape (qt, N, K) on the target, the slot
        # MUST remain empty so a draft model with the same shape gets a
        # chance to profile it. (Pre-fix, the target's exception path wrote
        # a fallback config that the draft loop would then skip.)
        import argparse

        target_gguf = tmp_path / "target.gguf"
        target_gguf.write_bytes(b"GGUF" + b"\x00" * 100)
        draft_gguf = tmp_path / "draft.gguf"
        draft_gguf.write_bytes(b"GGUF" + b"\x00" * 100)

        shape = ("Q4_K", 4096, 4096)
        target_profile = self._make_profile("T", [shape])
        draft_profile = self._make_profile("D", [shape])

        call_count = {"n": 0}
        def flaky_tune(*, N, K, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated transient failure on target")
            return ({"nwarps": 8, "rows_per_block": 4}, 100.0, 1.5)

        args = argparse.Namespace(
            gguf=str(target_gguf),
            output="smithy-config.h",
            max_configs=3,
            warmup=1,
            runs=2,
            no_bench=False,  # bench mode so _tune_shape_cli is invoked
            draft=[str(draft_gguf)],
        )

        from kernel_anvil.cli import cmd_gguf_optimize
        with patch("kernel_anvil.cli.parse_gguf", side_effect=[target_profile, draft_profile]), \
             patch("kernel_anvil.cli._tune_shape_cli", side_effect=flaky_tune), \
             patch("kernel_anvil.cli._get_gpu_spec") as gs, \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.device"):
            from kernel_anvil.rdna3 import GFX1100
            gs.return_value = GFX1100
            cmd_gguf_optimize(args)

        # Must have called _tune_shape_cli twice: once for target (failed)
        # then once for draft (recovered). If the target's failure had
        # poisoned the slot, the draft would have skipped it and call_count
        # would still be 1.
        assert call_count["n"] == 2

        cache = Path.home() / ".cache" / "smithy" / f"{Path(target_gguf).stem}.json"
        import json
        data = json.loads(cache.read_text())
        type_idx = "12"  # Q4_K
        # Bucket (3, 3) for (4096, 4096): N=4096 -> bucket 2, K=4096 -> bucket 2.
        cells = data["configs"].get(type_idx, {})
        assert cells.get("2,2") == {"nwarps": 8, "rows_per_block": 4}
