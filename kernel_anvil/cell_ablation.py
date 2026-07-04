"""Ground-truth cell ablation for the llama.cpp MMVQ runtime target.

Why this exists (2026-07-03): the proxy pipeline sweeps Triton configs and
maps BLOCK_N//64 onto rows_per_block -- but the patched llama.cpp runtime
can honor exactly ONE bit per (quant, N-bucket, K-bucket) cell: force the
small_k dispatch path on, or leave the default. nwarps in the JSON is dead
weight at runtime (it is __launch_bounds__-bound at compile time). So for
this target the only honest optimizer is: decide that one bit per cell by
measuring THE REAL MODEL on THE REAL KERNEL.

This module does that. It enumerates cells whose shapes fire upstream's own
small_k trigger formula (the same arithmetic mmvq.cu computes, which the
RDNA blanket-disable then discards), then A/Bs each candidate cell alone
against a baseline via short llama-bench decode runs, and emits a config
containing only the cells that measurably win.

The trigger math doubles as the honest --no-bench heuristic: "cells where
upstream's own formula wanted small_k but the vendor ban said no".
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from kernel_anvil.codegen import GGML_TYPE_MAP, bucket_index

# Per-quant constants from ggml-common.h / vecdotq.cuh (qk, qi, vdr_mmvq).
# qi = qk / (4 * qr). Only types that appear in decode GEMVs need entries;
# unknown types simply never become candidates.
QUANT_CONSTANTS: dict[str, tuple[int, int, int]] = {
    # quant: (qk, qi, vdr)
    "Q4_0": (32, 4, 2),
    "Q4_1": (32, 4, 2),
    "Q5_0": (32, 4, 2),
    "Q5_1": (32, 4, 2),
    "Q8_0": (32, 8, 2),
    "Q2_K": (256, 16, 1),
    "Q3_K": (256, 32, 1),
    "Q4_K": (256, 32, 2),
    "Q5_K": (256, 32, 2),
    "Q6_K": (256, 32, 1),
    "IQ4_NL": (32, 4, 2),
    "IQ4_XS": (256, 32, 2),
}

# calc_nwarps whitelists per upstream mmvq.cu (ncols_dst == 1). Types not
# listed get nwarps=1, which can never fire the trigger (nwarps > 1 gate).
NWARPS_WHITELIST: dict[str, dict[str, int]] = {
    "rdna3_0": {
        "Q4_0": 8, "Q4_1": 8, "Q5_0": 8, "Q5_1": 8,
        "Q8_0": 8, "Q4_K": 8, "Q6_K": 8, "IQ4_NL": 8,
    },
    "rdna4": {
        "Q4_0": 8, "Q4_1": 8, "Q5_0": 8, "Q5_1": 8, "Q8_0": 8,
        "Q2_K": 8, "Q4_K": 8, "Q5_K": 8, "Q6_K": 8,
        "IQ4_NL": 8, "IQ4_XS": 8,
    },
}

WARP_SIZE = 32  # RDNA wave32 as reported to ggml


def small_k_nwarps(quant: str, table: str = "rdna3_0") -> int:
    return NWARPS_WHITELIST.get(table, {}).get(quant, 1)


def small_k_trigger_fires(quant: str, k: int, table: str = "rdna3_0") -> bool:
    """Exact mirror of mmvq.cu's should_use_small_k arithmetic (ncols_dst=1).

    use = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp
    with blocks_per_row_x = K / qk (integer) and
    blocks_per_iter_1warp = vdr * warp_size / qi.
    """
    consts = QUANT_CONSTANTS.get(quant)
    if consts is None:
        return False
    qk, qi, vdr = consts
    nwarps = small_k_nwarps(quant, table)
    if nwarps <= 1:
        return False
    blocks_per_row = k // qk
    blocks_per_iter_1warp = vdr * WARP_SIZE // qi
    return blocks_per_row < nwarps * blocks_per_iter_1warp


@dataclass
class CandidateCell:
    quant: str
    n_bucket: int
    k_bucket: int
    shapes: list[tuple[int, int, int]] = field(default_factory=list)  # (N, K, count)

    @property
    def key(self) -> str:
        return f"{self.quant}:{self.n_bucket},{self.k_bucket}"


def candidate_cells(
    unique_shapes: dict[tuple[str, int, int], int],
    table: str = "rdna3_0",
) -> list[CandidateCell]:
    """Bucket-dedupe the shapes whose K fires the small_k trigger."""
    cells: dict[tuple[str, int, int], CandidateCell] = {}
    for (quant, n, k), count in sorted(unique_shapes.items()):
        if quant not in GGML_TYPE_MAP:
            continue
        if not small_k_trigger_fires(quant, k, table):
            continue
        cell_key = (quant, bucket_index(n), bucket_index(k))
        cell = cells.get(cell_key)
        if cell is None:
            cell = CandidateCell(quant, cell_key[1], cell_key[2])
            cells[cell_key] = cell
        cell.shapes.append((n, k, count))
    return list(cells.values())


# --- config file emission ---------------------------------------------------

# Baseline decoy: the loader chain falls through to default.json when a
# config file loads ZERO cells (SMITHY_CONFIG -> model stem -> default.json),
# so an "empty" baseline would silently inherit whatever default.json holds.
# Instead the baseline carries one cell for a type/bucket combination no
# real model dispatches (Q5_1 at the largest N/K buckets), which loads
# cleanly (loaded=1) and engages nothing.
DECOY_QUANT = "Q5_1"
DECOY_CELL = (4, 4)


def _config_json(cells: list[CandidateCell], table: str) -> dict:
    configs: dict[str, dict[str, dict[str, int]]] = {}
    for cell in cells:
        type_idx = str(GGML_TYPE_MAP[cell.quant])
        nwarps = small_k_nwarps(cell.quant, table)
        configs.setdefault(type_idx, {})[f"{cell.n_bucket},{cell.k_bucket}"] = {
            # rows_per_block > 1 is the runtime's actual decision bit; the
            # value mirrors what calc_rows_per_block will really use
            # (nwarps) rather than a proxy-derived number.
            "nwarps": nwarps,
            "rows_per_block": nwarps,
        }
    if not configs:
        decoy_idx = str(GGML_TYPE_MAP[DECOY_QUANT])
        configs[decoy_idx] = {
            f"{DECOY_CELL[0]},{DECOY_CELL[1]}": {"nwarps": 2, "rows_per_block": 2}
        }
    return {"version": 1, "generator": "kernel-anvil ablate", "configs": configs}


def write_config(path: str | Path, cells: list[CandidateCell], table: str) -> None:
    payload = json.dumps(_config_json(cells, table), indent=2)
    tmp = Path(str(path) + ".tmp")
    try:
        tmp.write_text(payload)
        os.rename(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


# --- llama-bench runner -------------------------------------------------------


@dataclass
class BenchResult:
    tokens_per_second: float
    raw: dict


def run_llama_bench(
    llama_bench: str,
    model: str,
    config_path: str,
    *,
    n_gen: int = 32,
    reps: int = 3,
    extra_args: list[str] | None = None,
    timeout: int = 900,
) -> BenchResult:
    """One llama-bench decode measurement under a given smithy config."""
    cmd = [
        llama_bench, "-m", model,
        "-p", "0", "-n", str(n_gen), "-r", str(reps),
        "-o", "json",
    ] + (extra_args or [])
    env = dict(os.environ)
    env["SMITHY_CONFIG"] = str(config_path)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-bench failed rc={proc.returncode}: {proc.stderr.strip()[-400:]}")
    data = json.loads(proc.stdout)
    rows = [r for r in data if r.get("n_gen", 0) > 0] if isinstance(data, list) else []
    if not rows:
        raise RuntimeError("llama-bench produced no tg rows")
    return BenchResult(tokens_per_second=float(rows[-1]["avg_ts"]), raw=rows[-1])


@dataclass
class AblationReport:
    baseline_ts: float
    cell_results: list[tuple[CandidateCell, float, float]]  # (cell, ts, ratio)
    winners: list[CandidateCell]
    config_path: str | None


def run_ablation(
    model: str,
    llama_bench: str,
    unique_shapes: dict[tuple[str, int, int], int],
    *,
    table: str = "rdna3_0",
    threshold: float = 1.01,
    n_gen: int = 32,
    reps: int = 3,
    extra_args: list[str] | None = None,
    out_path: str | Path | None = None,
    bench_fn=None,
    log=print,
) -> AblationReport:
    """A/B every candidate cell alone against a decoy baseline.

    bench_fn is injectable for tests: (config_path) -> BenchResult.
    """
    if bench_fn is None:
        def bench_fn(cfg_path: str) -> BenchResult:  # pragma: no cover
            return run_llama_bench(
                llama_bench, model, cfg_path,
                n_gen=n_gen, reps=reps, extra_args=extra_args)

    cells = candidate_cells(unique_shapes, table)
    if not cells:
        log("no candidate cells: no shape fires the small_k trigger for this table")
        return AblationReport(0.0, [], [], None)

    workdir = Path(tempfile.mkdtemp(prefix="anvil-ablate-"))

    baseline_cfg = workdir / "baseline.json"
    write_config(baseline_cfg, [], table)  # decoy-only
    log(f"[ablate] baseline (decoy config) ...")
    baseline = bench_fn(str(baseline_cfg))
    log(f"[ablate] baseline: {baseline.tokens_per_second:.2f} tok/s")

    results: list[tuple[CandidateCell, float, float]] = []
    for i, cell in enumerate(cells, 1):
        cfg = workdir / f"cell-{i}.json"
        write_config(cfg, [cell], table)
        r = bench_fn(str(cfg))
        ratio = (r.tokens_per_second / baseline.tokens_per_second
                 if baseline.tokens_per_second > 0 else 0.0)
        shapes = ", ".join(f"({n},{k})x{c}" for n, k, c in cell.shapes)
        log(f"[ablate] {i}/{len(cells)} {cell.key} [{shapes}]: "
            f"{r.tokens_per_second:.2f} tok/s ({ratio:.3f}x)")
        results.append((cell, r.tokens_per_second, ratio))

    winners = [cell for cell, _, ratio in results if ratio >= threshold]
    config_path = None
    if out_path is not None:
        write_config(out_path, winners, table)
        config_path = str(out_path)
        log(f"[ablate] wrote {len(winners)} winning cell(s) -> {out_path}")
    if not winners:
        log("[ablate] no cell beat the baseline by the threshold; "
            "config contains decoy only (engages nothing)")
    return AblationReport(baseline.tokens_per_second, results, winners, config_path)
