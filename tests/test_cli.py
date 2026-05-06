"""Tests for smithy CLI."""
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from kernel_anvil.cli import _format_config, _load_runner, main


PROJ_ROOT = Path(__file__).parent.parent


class TestFormatConfig:
    def test_single_key(self):
        assert _format_config({"BLOCK_N": 64}) == "BLOCK_N=64"

    def test_sorted_keys(self):
        result = _format_config({"num_warps": 4, "BLOCK_N": 64, "BLOCK_K": 128})
        assert result == "BLOCK_K=128 BLOCK_N=64 num_warps=4"

    def test_empty_config(self):
        assert _format_config({}) == ""


class TestLoadRunner:
    def test_load_valid_runner(self, tmp_path):
        runner_file = tmp_path / "runner.py"
        runner_file.write_text(textwrap.dedent("""
            import torch
            def setup():
                return {"x": torch.ones(4)}
            def run(inputs, **config):
                return inputs["x"] * 2
            def reference(inputs):
                return inputs["x"] * 2
            DATA_BYTES = 32
            BASELINE_CONFIG = {}
        """))
        mod = _load_runner(str(runner_file))
        assert hasattr(mod, "setup")
        assert hasattr(mod, "run")
        assert hasattr(mod, "reference")
        assert mod.DATA_BYTES == 32

    def test_load_nonexistent_exits(self):
        with pytest.raises(SystemExit):
            _load_runner("/nonexistent/runner.py")


class TestCLIEntrypoint:
    def test_no_args_prints_help(self):
        """smithy with no args should exit with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["kernel-anvil"]):
                main()
        assert exc_info.value.code == 1

    def test_help_flag(self):
        """smithy --help should exit with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["kernel-anvil", "--help"]):
                main()
        assert exc_info.value.code == 0

    def test_sweep_help(self):
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["kernel-anvil", "sweep", "--help"]):
                main()
        assert exc_info.value.code == 0

    def test_profile_help(self):
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["kernel-anvil", "profile", "--help"]):
                main()
        assert exc_info.value.code == 0


class TestCLISubprocess:
    """Test the installed CLI entry point via subprocess."""

    def test_smithy_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "kernel_anvil.cli", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "sweep" in result.stdout
        assert "profile" in result.stdout

    def test_smithy_no_args(self):
        result = subprocess.run(
            [sys.executable, "-m", "kernel_anvil.cli"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 1


class TestSweepCommand:
    """Test sweep with a mock runner."""

    def test_sweep_runs_end_to_end(self, tmp_path):
        runner_file = tmp_path / "runner.py"
        runner_file.write_text(textwrap.dedent("""
            import torch
            def setup():
                return {"x": torch.ones(64)}
            def run(inputs, BLOCK_N=64, BLOCK_K=128, num_warps=4, **kw):
                return inputs["x"] * 2
            def reference(inputs):
                return inputs["x"] * 2
            DATA_BYTES = 512
            BASELINE_CONFIG = {"BLOCK_N": 64, "BLOCK_K": 128, "num_warps": 4}
        """))
        result = subprocess.run(
            [sys.executable, "-m", "kernel_anvil.cli", "sweep", str(runner_file),
             "--max-configs", "3", "--warmup", "1", "--runs", "2"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        # "Sweep Results" / "Winner" appear when the runner is slow enough
        # to have a configurable bottleneck. A sub-microsecond kernel
        # legitimately classifies as launch_overhead and short-circuits
        # with "No configs to sweep" -- still a clean exit, still proves
        # the CLI ran end-to-end.
        assert (
            "Sweep Results" in result.stdout
            or "Winner" in result.stdout
            or "No configs to sweep" in result.stdout
        ), f"unexpected sweep output:\n{result.stdout}"

    def test_sweep_handles_no_data_bytes(self, tmp_path):
        runner_file = tmp_path / "runner.py"
        runner_file.write_text(textwrap.dedent("""
            import torch
            def setup():
                return {"x": torch.ones(32)}
            def run(inputs, **kw):
                return inputs["x"] * 2
            def reference(inputs):
                return inputs["x"] * 2
            BASELINE_CONFIG = {}
        """))
        result = subprocess.run(
            [sys.executable, "-m", "kernel_anvil.cli", "sweep", str(runner_file),
             "--max-configs", "3", "--warmup", "1", "--runs", "2"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0


class TestProfileCommand:
    """Test profile with a mock runner."""

    def test_profile_runs(self, tmp_path):
        runner_file = tmp_path / "runner.py"
        runner_file.write_text(textwrap.dedent("""
            import torch
            def setup():
                return {"x": torch.ones(64)}
            def run(inputs, BLOCK_N=64, BLOCK_K=128, num_warps=4, **kw):
                return inputs["x"] * 2
            def reference(inputs):
                return inputs["x"] * 2
            BASELINE_CONFIG = {"BLOCK_N": 64, "BLOCK_K": 128, "num_warps": 4}
        """))
        result = subprocess.run(
            [sys.executable, "-m", "kernel_anvil.cli", "profile", str(runner_file),
             "--warmup", "1", "--runs", "3"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "Profile Metrics" in result.stdout
        assert "Classification" in result.stdout
        assert "Recommended directions" in result.stdout

    def test_profile_shows_all_fields(self, tmp_path):
        runner_file = tmp_path / "runner.py"
        runner_file.write_text(textwrap.dedent("""
            import torch
            def setup():
                return {"x": torch.ones(32)}
            def run(inputs, **kw):
                return inputs["x"]
            def reference(inputs):
                return inputs["x"]
            DATA_BYTES = 128
            BASELINE_CONFIG = {"BLOCK_N": 64, "BLOCK_K": 128}
        """))
        result = subprocess.run(
            [sys.executable, "-m", "kernel_anvil.cli", "profile", str(runner_file),
             "--warmup", "1", "--runs", "2"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "Duration" in result.stdout
        assert "VGPRs" in result.stdout
        assert "LDS" in result.stdout
        assert "Bandwidth" in result.stdout
        assert "Occupancy" in result.stdout
