"""Sweep llama.cpp MMVQ kernel configs using actual llama-bench + rocprofv3.

Instead of profiling a Triton proxy kernel, this runs llama-bench with different
smithy configs and uses rocprofv3 to measure individual kernel times. This gives
accurate per-shape timings on the ACTUAL production kernel.

The sweep:
1. Parse GGUF to identify unique (quant_type, N, K) shapes
2. For each candidate nwarps (1, 2, 4, 8):
   a. Generate a smithy config with that nwarps for ALL shapes
   b. Run llama-bench with rocprofv3 kernel tracing
   c. Parse kernel trace to get per-shape timings
3. For each shape, pick the nwarps that gave the lowest time
4. Write the optimal per-shape config to ~/.cache/smithy/
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from kernel_anvil.codegen import GGML_TYPE_MAP, bucket_index
from kernel_anvil.gguf import parse_gguf


@dataclass
class KernelTiming:
    """Timing for a specific kernel invocation pattern."""
    kernel_name: str
    quant_type: int  # ggml_type enum value
    calls: int
    total_us: float
    avg_us: float
    grid_x: int  # approximates N/rows_per_block


@dataclass
class SweepResult:
    """Result of a full nwarps sweep for one model."""
    model_path: str
    gpu: str
    shape_configs: dict  # (type_idx, n_bucket, k_bucket) -> {"nwarps": int, "speedup": float}
    baseline_tps: float
    optimized_tps: float


def _validate_sql_identifier(name: str) -> bool:
    """Validate that a string is a safe SQL identifier (table/column name).

    Only allows alphanumeric characters and underscores, starting with a
    letter or underscore. Rejects everything else -- no quoting tricks,
    no semicolons, no spaces.
    """
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name))


def _parse_rocprof_db(db_path: str) -> list[KernelTiming]:
    """Parse rocprofv3 SQLite database for MMVQ kernel timings."""
    db = sqlite3.connect(db_path)

    # Find the UUID suffix for this run's tables
    tables = [r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'rocpd_kernel_dispatch_%'"
    ).fetchall()]

    if not tables:
        db.close()
        return []

    dispatch_table = tables[0]

    # Table names come from an external rocprofv3 database -- untrusted input.
    # Validate both table names as safe SQL identifiers before interpolation.
    if not _validate_sql_identifier(dispatch_table):
        db.close()
        return []

    uuid = dispatch_table.replace("rocpd_kernel_dispatch_", "")
    symbol_table = f"rocpd_info_kernel_symbol_{uuid}"

    if not _validate_sql_identifier(symbol_table):
        db.close()
        return []

    rows = db.execute(f"""
        SELECT ks.kernel_name,
               COUNT(*) as calls,
               SUM(d.end - d.start) / 1000.0 as total_us,
               AVG(d.end - d.start) / 1000.0 as avg_us,
               d.grid_size_x
        FROM {dispatch_table} d
        JOIN {symbol_table} ks ON d.kernel_id = ks.id
        WHERE ks.kernel_name LIKE '%mul_mat_vec_q%'
        GROUP BY ks.kernel_name, d.grid_size_x
        ORDER BY total_us DESC
    """).fetchall()

    timings = []
    for name, calls, total_us, avg_us, grid_x in rows:
        # Extract quant type from kernel name: mul_mat_vec_q<(ggml_type)12, ...>
        qt = -1
        if "(ggml_type)" in name:
            try:
                qt = int(name.split("(ggml_type)")[1].split(",")[0].split(")")[0])
            except (ValueError, IndexError):
                pass

        timings.append(KernelTiming(
            kernel_name=name, quant_type=qt, calls=calls,
            total_us=total_us, avg_us=avg_us, grid_x=grid_x,
        ))

    db.close()
    return timings


def _run_bench_with_config(
    llama_bench: str,
    model_path: str,
    config_path: str,
    prompt_tokens: int = 16,
    gen_tokens: int = 32,
) -> tuple[float, list[KernelTiming]]:
    """Run llama-bench with rocprofv3 and return (tok/s, kernel timings)."""

    with tempfile.TemporaryDirectory() as tmpdir:
        env = os.environ.copy()
        env["SMITHY_CONFIG"] = config_path

        # Run with rocprofv3 kernel trace
        cmd = [
            "rocprofv3", "--kernel-trace",
            "--output-directory", tmpdir,
            "--",
            llama_bench,
            "-m", model_path,
            "-ngl", "999",
            "-t", "12",
            "-p", str(prompt_tokens),
            "-n", str(gen_tokens),
            "-r", "1",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, env=env,
        )

        # Extract tok/s from llama-bench output
        tps = 0.0
        for line in result.stdout.split("\n"):
            if f"tg{gen_tokens}" in line:
                parts = line.split("|")
                for p in parts:
                    p = p.strip()
                    try:
                        val = float(p.split("±")[0].strip())
                        if val > 1.0:  # tok/s will be > 1
                            tps = val
                    except (ValueError, IndexError):
                        pass

        # Find the rocprofv3 results database
        timings = []
        for f in Path(tmpdir).rglob("*results.db"):
            timings = _parse_rocprof_db(str(f))
            break

        return tps, timings


def _gen_config(nwarps: int, shapes: dict) -> str:
    """Generate a smithy JSON config with uniform nwarps for all shapes."""
    configs: dict[str, dict] = {}

    for (qt_name, n, k), count in shapes.items():
        type_idx = GGML_TYPE_MAP.get(qt_name)
        if type_idx is None:
            continue

        ni = bucket_index(n)
        ki = bucket_index(k)
        key = str(type_idx)

        if key not in configs:
            configs[key] = {}

        configs[key][f"{ni},{ki}"] = {
            "nwarps": nwarps,
            "rows_per_block": 1,
        }

    return json.dumps({"gpu": "auto", "model": "sweep", "configs": configs}, indent=2)


def sweep_model(
    model_path: str,
    llama_bench: str | None = None,
    nwarps_candidates: list[int] | None = None,
    verbose: bool = True,
) -> SweepResult:
    """Sweep nwarps values on actual llama.cpp kernels for a GGUF model.

    Args:
        model_path: Path to GGUF model file.
        llama_bench: Path to llama-bench binary. Auto-detected if None.
        nwarps_candidates: List of nwarps values to try. Default [1, 2, 4, 8].
        verbose: Print progress.

    Returns:
        SweepResult with optimal per-shape configs.
    """
    if nwarps_candidates is None:
        nwarps_candidates = [1, 2, 4, 8]

    if llama_bench is None:
        # Try common locations
        for p in [
            Path.home() / "Projects/llama-cpp-turboquant/build/bin/llama-bench",
            Path("/usr/local/bin/llama-bench"),
        ]:
            if p.exists():
                llama_bench = str(p)
                break
        if llama_bench is None:
            raise FileNotFoundError("llama-bench not found. Pass --llama-bench path.")

    # Parse GGUF for shapes
    if verbose:
        print(f"Parsing {Path(model_path).name}...")
    profile = parse_gguf(model_path)
    shapes = profile.unique_shapes

    if verbose:
        print(f"Found {len(shapes)} unique GEMV shapes")

    # Run baseline (no smithy config)
    if verbose:
        print(f"\nBaseline (stock llama.cpp)...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        # Empty config = no overrides
        f.write('{"configs": {}}')
        empty_config = f.name

    try:
        baseline_tps, baseline_timings = _run_bench_with_config(
            llama_bench, model_path, empty_config,
        )
    finally:
        os.unlink(empty_config)

    if verbose:
        print(f"  Baseline: {baseline_tps:.2f} tok/s")

    # Sweep each nwarps value
    results_by_nwarps: dict[int, tuple[float, list[KernelTiming]]] = {}

    for nw in nwarps_candidates:
        if verbose:
            print(f"  Sweeping nwarps={nw}...", end=" ", flush=True)

        config_json = _gen_config(nw, shapes)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(config_json)
            config_path = f.name

        try:
            tps, timings = _run_bench_with_config(llama_bench, model_path, config_path)
            results_by_nwarps[nw] = (tps, timings)
            if verbose:
                print(f"{tps:.2f} tok/s")
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
        finally:
            os.unlink(config_path)

    # Find the best global nwarps (simple approach)
    if not results_by_nwarps:
        raise RuntimeError("All nwarps benchmark attempts failed. Check GPU and llama-bench.")
    best_nwarps = max(results_by_nwarps, key=lambda nw: results_by_nwarps[nw][0])
    best_tps = results_by_nwarps[best_nwarps][0]

    if verbose:
        speedup_pct = (best_tps / baseline_tps - 1) * 100 if baseline_tps > 0 else 0
        print(f"\n  Best: nwarps={best_nwarps} at {best_tps:.2f} tok/s "
              f"(+{speedup_pct:.1f}% vs baseline)")

    # Build per-shape config using the best global nwarps
    # TODO: per-shape optimization using rocprofv3 kernel timings
    shape_configs = {}
    for (qt_name, n, k), count in shapes.items():
        type_idx = GGML_TYPE_MAP.get(qt_name)
        if type_idx is None:
            continue
        ni = bucket_index(n)
        ki = bucket_index(k)
        shape_configs[(type_idx, ni, ki)] = {
            "nwarps": best_nwarps,
            "rows_per_block": 1,
        }

    return SweepResult(
        model_path=model_path,
        gpu="auto",
        shape_configs=shape_configs,
        baseline_tps=baseline_tps,
        optimized_tps=best_tps,
    )


def write_optimal_config(result: SweepResult, cache_dir: str | None = None):
    """Write the optimal config to ~/.cache/smithy/ for llama.cpp auto-loading."""
    cache_dir = cache_dir or os.path.expanduser("~/.cache/smithy")
    os.makedirs(cache_dir, exist_ok=True)

    model_name = Path(result.model_path).stem
    config = {"gpu": result.gpu, "model": model_name, "configs": {}}

    for (type_idx, ni, ki), cfg in result.shape_configs.items():
        key = str(type_idx)
        if key not in config["configs"]:
            config["configs"][key] = {}
        config["configs"][key][f"{ni},{ki}"] = cfg

    path = os.path.join(cache_dir, f"{model_name}.json")

    # Atomic write
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(config, f, indent=2)
        os.rename(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise

    return path
