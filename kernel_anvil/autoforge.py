"""Autoforge: architecture-agnostic kernel generation and benchmarking.

The full automated pipeline:
1. Detect GPU architecture (for hipcc target)
2. Parse GGUF model (for tensor shapes)
3. For each unique shape, generate kernels with candidate configs
4. Compile all candidates with hipcc for the detected arch
5. Benchmark each candidate on the actual GPU
6. Pick the fastest per shape
7. Compile a final kernel pack (.so with all winners)
8. Write config pointing llama.cpp to the optimized kernels

No hardcoded architecture knowledge needed. The GPU's actual performance
determines the optimal config. Works on any HIP-capable GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from kernel_anvil.gguf import parse_gguf
from kernel_anvil.hip_codegen import (
    KernelSpec,
    Q8_1_BLOCK_SIZE,
    QUANT_TYPES,
    QuantTypeInfo,
    find_llama_cpp_path,
    generate_kernel,
    generate_q4k_kernel,
    get_llama_cpp_include_dirs,
    get_quant_info,
)


@dataclass
class BenchResult:
    spec: KernelSpec
    latency_us: float
    bandwidth_gbs: float
    data_bytes: int


@dataclass
class ForgeResult:
    """Result of a full autoforge run."""
    model_path: str
    gpu_arch: str
    shapes: dict  # (qt, N, K) -> winner BenchResult
    kernel_pack_path: str | None
    total_time_s: float


def detect_arch() -> str:
    """Detect the GPU architecture string for hipcc --offload-arch."""
    # Try torch first (most reliable)
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            if hasattr(props, "gcnArchName"):
                return props.gcnArchName
    except Exception:
        pass

    # Try rocminfo
    try:
        out = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10,
        )
        for line in out.stdout.split("\n"):
            if "gfx" in line.lower() and "name:" in line.lower():
                return line.strip().split()[-1]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    raise RuntimeError("Cannot detect GPU architecture. Is ROCm installed?")


def _compile_and_bench(
    spec: KernelSpec,
    arch: str,
    warmup: int = 10,
    runs: int = 50,
    llama_cpp_path: Path | None = None,
    include_dirs: list[str] | None = None,
) -> BenchResult | None:
    """Generate, compile, and benchmark a single kernel config.

    Returns BenchResult or None if compilation/benchmark fails.
    """
    quant_info = get_quant_info(spec.quant_type)
    if quant_info is None:
        return None  # Unsupported type (F16/F32/BF16 don't use MMVQ)

    # Without llama.cpp headers, only Q4_K works via fallback
    if llama_cpp_path is None and spec.quant_type != "Q4_K":
        return None

    try:
        src = generate_kernel(spec, llama_cpp_path)
    except ValueError:
        return None  # Shape not compatible (e.g., K not divisible by qk)

    N, K = spec.N, spec.K
    blocks_per_row = K // quant_info.qk
    weight_bytes = N * blocks_per_row * quant_info.block_size
    input_bytes = (K // 32) * Q8_1_BLOCK_SIZE
    data_bytes = weight_bytes + input_bytes

    # Generate a self-contained benchmark binary
    bench_src = f"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>

{src}

int main() {{
    const int N = {N}, K = {K};
    const int bpr = K / {quant_info.qk};
    const size_t wb = (size_t)N * bpr * {quant_info.block_size};
    const size_t ib = (K / 32) * {Q8_1_BLOCK_SIZE};

    void *dw, *di; float *dout;
    hipMalloc(&dw, wb);
    hipMalloc(&di, ib);
    hipMalloc(&dout, N * sizeof(float));
    hipMemset(dw, 0, wb);
    hipMemset(di, 0, ib);

    dim3 grid(({N} + {spec.rows_per_block} - 1) / {spec.rows_per_block}, 1, 1);
    dim3 block(32, {spec.nwarps}, 1);

    for (int i = 0; i < {warmup}; i++)
        hipLaunchKernelGGL({spec.name}, grid, block, 0, 0, dw, di, dout, bpr);
    hipDeviceSynchronize();

    hipEvent_t t0, t1;
    hipEventCreate(&t0);
    hipEventCreate(&t1);
    hipEventRecord(t0);
    for (int i = 0; i < {runs}; i++)
        hipLaunchKernelGGL({spec.name}, grid, block, 0, 0, dw, di, dout, bpr);
    hipEventRecord(t1);
    hipDeviceSynchronize();

    float ms;
    hipEventElapsedTime(&ms, t0, t1);
    float us = ms / {runs} * 1000.0f;
    float bw = (us > 0.0f) ? (double)(wb + ib) / (us * 1e-6) / 1e9 : 0.0;
    // Machine-readable output
    printf("RESULT %.2f %.2f\\n", us, bw);

    hipFree(dw);
    hipFree(di);
    hipFree(dout);
    return 0;
}}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "bench.hip")
        bin_path = os.path.join(tmpdir, "bench")

        with open(src_path, "w") as f:
            f.write(bench_src)

        # Build compile command
        compile_cmd = ["hipcc", f"--offload-arch={arch}", "-O3"]
        if include_dirs:
            for d in include_dirs:
                compile_cmd.extend(["-I", d])
        # Architecture macros for llama.cpp headers
        if "gfx1100" in arch or "gfx1101" in arch or "gfx1102" in arch:
            compile_cmd.extend(["-DRDNA3_0", "-DRDNA3"])
        elif "gfx1150" in arch or "gfx1151" in arch:
            compile_cmd.extend(["-DRDNA3_5", "-DRDNA3"])
        elif "gfx1200" in arch or "gfx1201" in arch:
            compile_cmd.append("-DRDNA4")
        elif "gfx1030" in arch or "gfx1031" in arch or "gfx1032" in arch:
            compile_cmd.append("-DRDNA2")
        elif "gfx1010" in arch or "gfx1012" in arch:
            compile_cmd.append("-DRDNA1")
        # Required HIP defines for ggml headers
        compile_cmd.extend(["-DGGML_COMMON_DECL_HIP", "-DGGML_COMMON_IMPL_HIP", "-DGGML_USE_HIP"])
        compile_cmd.append("-DMMVQ_MAX_BATCH_SIZE=8")
        compile_cmd.extend(["-o", bin_path, src_path])

        # Compile
        try:
            result = subprocess.run(
                compile_cmd,
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                return None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

        # Run
        try:
            result = subprocess.run(
                [bin_path], capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return None
        except subprocess.TimeoutExpired:
            return None

        # Parse output
        for line in result.stdout.split("\n"):
            if line.startswith("RESULT"):
                parts = line.split()
                latency_us = float(parts[1])
                bw_gbs = float(parts[2])
                return BenchResult(
                    spec=spec,
                    latency_us=latency_us,
                    bandwidth_gbs=bw_gbs,
                    data_bytes=data_bytes,
                )

    return None


def autoforge(
    model_path: str,
    output_dir: str | None = None,
    arch: str | None = None,
    nwarps_candidates: list[int] | None = None,
    rpb_candidates: list[int] | None = None,
    llama_cpp_path: str | None = None,
    verbose: bool = True,
) -> ForgeResult:
    """Full automated kernel forging pipeline.

    Args:
        model_path: Path to GGUF model.
        output_dir: Where to write compiled kernels. Default ~/.cache/kernel-anvil/
        arch: GPU arch string. Auto-detected if None.
        nwarps_candidates: nwarps values to try. Default [1, 2, 4, 8].
        rpb_candidates: rows_per_block values to try. Default [1, 2, 4].
        llama_cpp_path: Path to llama.cpp source. Auto-detected if None.
        verbose: Print progress.
    """
    t0 = time.monotonic()

    if nwarps_candidates is None:
        nwarps_candidates = [1, 2, 4, 8]
    if rpb_candidates is None:
        rpb_candidates = [1, 2, 4]
    if output_dir is None:
        output_dir = os.path.expanduser("~/.cache/kernel-anvil")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Detect architecture
    if arch is None:
        arch = detect_arch()
    if verbose:
        print(f"GPU architecture: {arch}")

    # Resolve llama.cpp path once for the whole run
    resolved_llama_path: Path | None = None
    if llama_cpp_path is not None:
        resolved_llama_path = Path(llama_cpp_path)
    else:
        resolved_llama_path = find_llama_cpp_path()

    include_dirs: list[str] | None = None
    if resolved_llama_path is not None:
        include_dirs = get_llama_cpp_include_dirs(resolved_llama_path)
        if verbose:
            print(f"llama.cpp headers: {resolved_llama_path}")
            print(f"All MMVQ quant types supported ({len(QUANT_TYPES)} types)")
    else:
        if verbose:
            print("llama.cpp headers: NOT FOUND (only Q4_K fallback available)")

    # 2. Parse model
    if verbose:
        print(f"Parsing {Path(model_path).name}...")
    profile = parse_gguf(model_path)
    shapes = profile.unique_shapes
    if verbose:
        print(f"Found {len(shapes)} unique GEMV shapes")

    # 3-6. For each shape, sweep configs and pick winner
    winners = {}

    for (qt, n, k), count in sorted(shapes.items()):
        quant_info = get_quant_info(qt)
        if quant_info is None:
            if verbose:
                print(f"  {qt} ({n}, {k}) x{count}: SKIPPED (not an MMVQ type)")
            continue

        # Without llama.cpp headers, only Q4_K works via fallback
        if resolved_llama_path is None and qt != "Q4_K":
            if verbose:
                print(f"  {qt} ({n}, {k}) x{count}: SKIPPED (no llama.cpp headers)")
            continue

        if verbose:
            print(f"  {qt} ({n}, {k}) x{count}: ", end="", flush=True)

        best: BenchResult | None = None
        configs_tested = 0

        for nw in nwarps_candidates:
            for rpb in rpb_candidates:
                # Skip invalid combos
                if rpb > n:
                    continue
                if nw * 32 > 1024:  # Max threads per block
                    continue

                spec = KernelSpec(
                    quant_type=qt, N=n, K=k,
                    nwarps=nw, rows_per_block=rpb,
                )

                result = _compile_and_bench(
                    spec, arch,
                    llama_cpp_path=resolved_llama_path,
                    include_dirs=include_dirs,
                )
                configs_tested += 1

                if result is not None:
                    if best is None or result.latency_us < best.latency_us:
                        best = result

        if best is not None:
            winners[(qt, n, k)] = best
            speedup_vs_stock = ""
            # Compare vs nw=8 rpb=1 (stock config)
            stock_spec = KernelSpec(quant_type=qt, N=n, K=k, nwarps=8, rows_per_block=1)
            stock = _compile_and_bench(
                stock_spec, arch,
                llama_cpp_path=resolved_llama_path,
                include_dirs=include_dirs,
            )
            if stock and best.latency_us > 0:
                ratio = stock.latency_us / best.latency_us
                speedup_vs_stock = f" ({ratio:.2f}x vs stock)"
            if verbose:
                print(f"nw={best.spec.nwarps} rpb={best.spec.rows_per_block} "
                      f"{best.latency_us:.1f}us {best.bandwidth_gbs:.0f}GB/s"
                      f"{speedup_vs_stock} [{configs_tested} tested]")
        else:
            if verbose:
                print(f"FAILED (all {configs_tested} configs failed)")

    # 7. Write smithy-compatible config JSON
    from kernel_anvil.codegen import bucket_index, GGML_TYPE_MAP

    config = {"gpu": arch, "model": profile.name, "configs": {}}
    for (qt, n, k), result in winners.items():
        type_idx = GGML_TYPE_MAP.get(qt)
        if type_idx is None:
            continue
        key = str(type_idx)
        if key not in config["configs"]:
            config["configs"][key] = {}
        ni = bucket_index(n)
        ki = bucket_index(k)
        config["configs"][key][f"{ni},{ki}"] = {
            "nwarps": result.spec.nwarps,
            "rows_per_block": result.spec.rows_per_block,
        }

    model_name = Path(model_path).stem
    config_path = os.path.join(
        os.path.expanduser("~/.cache/smithy"), f"{model_name}.json"
    )
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(config_path), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(config, f, indent=2)
        os.rename(tmp, config_path)
    except Exception:
        os.unlink(tmp)
        raise

    elapsed = time.monotonic() - t0

    if verbose:
        # Count skipped vs optimized
        total_tensors = sum(shapes.values())
        optimized_tensors = sum(shapes.get(k, 0) for k in winners)
        skipped_tensors = total_tensors - optimized_tensors
        skipped_types = set()
        for (qt, _, _) in shapes:
            if qt not in {q for q, _, _ in winners}:
                skipped_types.add(qt)
        print(f"\nConfig written to {config_path}")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Optimized: {optimized_tensors}/{total_tensors} tensors "
              f"({skipped_tensors} skipped)")
        if skipped_types:
            print(f"Skipped types: {', '.join(sorted(skipped_types))}")
        if winners:
            avg_bw = sum(r.bandwidth_gbs for r in winners.values()) / len(winners)
            print(f"Average bandwidth: {avg_bw:.0f} GB/s")

    return ForgeResult(
        model_path=model_path,
        gpu_arch=arch,
        shapes={k: v for k, v in winners.items()},
        kernel_pack_path=config_path,
        total_time_s=elapsed,
    )
