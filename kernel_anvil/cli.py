"""CLI for kernel-anvil -- profile-guided Triton kernel optimizer."""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

from kernel_anvil.analyze import classify
from kernel_anvil.codegen import generate_config_header
from kernel_anvil.gguf import parse_gguf, print_model_summary
from kernel_anvil.profile import profile_kernel
from kernel_anvil.rdna3 import detect_gpu, GFX1100
from kernel_anvil.sweep import generate_configs
from kernel_anvil.verify import verify_and_bench


console = Console()


def _load_runner(path: str):
    """Import a runner script as a module."""
    p = Path(path).resolve()
    if not p.exists():
        console.print(f"[red]Runner not found: {p}[/red]")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("runner", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_gpu_spec():
    """Detect GPU or fall back to GFX1100."""
    gpu = detect_gpu()
    if gpu is None:
        console.print("[yellow]No RDNA3 GPU detected, using GFX1100 defaults[/yellow]")
        return GFX1100
    console.print(f"[green]Detected: {gpu.name} ({gpu.gfx})[/green]")
    return gpu


def _format_config(cfg: dict) -> str:
    """Compact config string for table display."""
    parts = []
    for k in sorted(cfg):
        parts.append(f"{k}={cfg[k]}")
    return " ".join(parts)


def cmd_sweep(args):
    """Run end-to-end optimization sweep."""
    runner = _load_runner(args.runner)
    gpu = _get_gpu_spec()

    # Setup
    inputs = runner.setup()
    ref_output = runner.reference(inputs)
    baseline_config = getattr(runner, "BASELINE_CONFIG", {})
    data_bytes = getattr(runner, "DATA_BYTES", None)

    console.print("\n[bold]Profiling baseline...[/bold]")

    def kernel_fn(config):
        return runner.run(inputs, **config)

    # Profile baseline
    metrics = profile_kernel(
        kernel_fn=kernel_fn,
        config=baseline_config,
        data_bytes=data_bytes,
        gpu_spec=gpu,
        warmup=args.warmup,
        runs=args.runs,
    )

    # Classify bottleneck
    report = classify(metrics, gpu)

    # Benchmark baseline for speedup reference
    baseline_result = verify_and_bench(
        kernel_fn=kernel_fn,
        reference_output=ref_output,
        config=baseline_config,
        warmup=args.warmup,
        runs=args.runs,
        atol=args.atol,
        rtol=args.rtol,
        data_bytes=data_bytes,
    )
    baseline_latency = baseline_result.latency_us

    console.print(f"Baseline latency: [cyan]{baseline_latency:.1f} us[/cyan]")
    console.print(f"Bottleneck: [yellow]{report.classification}[/yellow] (severity {report.severity:.2f})")

    # Generate configs
    configs = generate_configs(
        report,
        baseline_config=baseline_config or None,
        max_configs=args.max_configs,
    )

    if not configs:
        console.print("[yellow]No configs to sweep (launch_overhead -- consider kernel fusion)[/yellow]")
        return

    console.print(f"\n[bold]Sweeping {len(configs)} configs...[/bold]")

    # Verify and benchmark each config
    results = []
    for i, cfg in enumerate(configs):
        try:
            result = verify_and_bench(
                kernel_fn=kernel_fn,
                reference_output=ref_output,
                config=cfg,
                warmup=args.warmup,
                runs=args.runs,
                atol=args.atol,
                rtol=args.rtol,
                data_bytes=data_bytes,
                baseline_latency_us=baseline_latency,
            )
            results.append(result)
        except Exception as e:
            console.print(f"  Config {i+1}/{len(configs)} failed: {e}")

    # Sort: correct results by latency (fastest first), then incorrect at the end
    correct = [r for r in results if r.correct]
    incorrect = [r for r in results if not r.correct]
    correct.sort(key=lambda r: r.latency_us)
    ranked = correct + incorrect

    # Print results table
    table = Table(title="Sweep Results")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Config", style="cyan")
    table.add_column("Latency (us)", justify="right")
    if data_bytes is not None:
        table.add_column("BW (GB/s)", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Status", justify="center")

    for i, r in enumerate(ranked):
        speedup_str = f"{r.speedup:.2f}x" if r.speedup is not None else "-"
        status = "[green]OK[/green]" if r.correct else "[red]FAIL[/red]"
        latency_str = f"{r.latency_us:.1f}"

        row = [str(i + 1), _format_config(r.config), latency_str]
        if data_bytes is not None:
            bw_str = f"{r.bandwidth_gbs:.1f}" if r.bandwidth_gbs is not None else "-"
            row.append(bw_str)
        row.extend([speedup_str, status])
        table.add_row(*row)

    console.print()
    console.print(table)

    # Print bottleneck info
    console.print(f"\n[bold]Bottleneck:[/bold] {report.classification}")
    console.print("[bold]Recommended directions:[/bold]")
    for d in report.directions:
        console.print(f"  - {d}")

    # Print winner
    if correct:
        winner = correct[0]
        console.print(f"\n[bold green]Winner:[/bold green] {_format_config(winner.config)}")
        console.print(f"  Latency: {winner.latency_us:.1f} us", end="")
        if winner.speedup is not None:
            console.print(f"  Speedup: {winner.speedup:.2f}x", end="")
        console.print()
    else:
        console.print("\n[red]No correct configs found.[/red]")


def cmd_profile(args):
    """Profile a kernel and classify its bottleneck."""
    runner = _load_runner(args.runner)
    gpu = _get_gpu_spec()

    inputs = runner.setup()
    baseline_config = getattr(runner, "BASELINE_CONFIG", {})
    data_bytes = getattr(runner, "DATA_BYTES", None)

    def kernel_fn(config):
        return runner.run(inputs, **config)

    console.print("[bold]Profiling...[/bold]")

    metrics = profile_kernel(
        kernel_fn=kernel_fn,
        config=baseline_config,
        data_bytes=data_bytes,
        gpu_spec=gpu,
        warmup=args.warmup,
        runs=args.runs,
    )

    report = classify(metrics, gpu)

    # Print metrics
    table = Table(title="Profile Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Duration", f"{metrics.duration_ns:.0f} ns ({metrics.duration_ns/1000:.1f} us)")
    table.add_row("VGPRs", str(metrics.vgpr_count))
    table.add_row("LDS", f"{metrics.lds_bytes} bytes")
    table.add_row("Scratch", f"{metrics.scratch_bytes} bytes")
    table.add_row("Bandwidth", f"{metrics.bandwidth_gbs:.1f} GB/s")
    table.add_row("Occupancy", f"{metrics.occupancy_pct:.1f}%")
    table.add_row("Limiting factor", metrics.limiting_factor)
    table.add_row("Threads/WG", str(metrics.threads_per_wg))
    console.print()
    console.print(table)

    # Print bottleneck report
    console.print(f"\n[bold]Classification:[/bold] {report.classification}")
    console.print(f"[bold]Severity:[/bold] {report.severity:.2f}")
    console.print(f"[bold]Limiting factor:[/bold] {report.limiting_factor}")
    console.print("[bold]Recommended directions:[/bold]")
    for d in report.directions:
        console.print(f"  - {d}")


def _make_runner(N: int, K: int, device: torch.device):
    """Create an on-the-fly GEMV runner for a given (N, K) shape.

    Returns (kernel_fn, reference_output, data_bytes).
    """
    from kernel_anvil.kernels import triton_gemv

    x = torch.randn(K, device=device, dtype=torch.float16)
    w = torch.randn(N, K, device=device, dtype=torch.float16)
    ref = F.linear(x.unsqueeze(0), w).squeeze(0)
    data_bytes = (N * K + K + N) * 2  # FP16 = 2 bytes

    def kernel_fn(config):
        return triton_gemv(x, w, bias=None, config=config)

    return kernel_fn, ref, data_bytes


def _tune_shape_cli(
    N: int,
    K: int,
    device: torch.device,
    gpu_spec,
    max_configs: int,
    warmup: int,
    runs: int,
) -> tuple[dict, float, float | None]:
    """Tune a single (N, K) shape and return (best_config, baseline_us, speedup).

    The best_config dict has keys: nwarps, rows_per_block (for codegen).
    """
    kernel_fn, ref, data_bytes = _make_runner(N, K, device)

    if gpu_spec is None:
        gpu_spec = GFX1100

    baseline_config = {"BLOCK_N": 64, "BLOCK_K": 128, "num_warps": 4, "num_stages": 1}

    # Profile baseline
    metrics = profile_kernel(
        kernel_fn=kernel_fn,
        config=baseline_config,
        data_bytes=data_bytes,
        gpu_spec=gpu_spec,
        warmup=warmup,
        runs=runs,
    )

    report = classify(metrics, gpu_spec)

    # Benchmark baseline
    baseline_result = verify_and_bench(
        kernel_fn=kernel_fn,
        reference_output=ref,
        config=baseline_config,
        warmup=warmup,
        runs=runs,
        data_bytes=data_bytes,
    )
    baseline_latency = baseline_result.latency_us

    # Generate candidates
    candidates = generate_configs(
        report,
        baseline_config=baseline_config,
        max_configs=max_configs,
    )

    best_config = baseline_config
    best_latency = baseline_latency

    for cfg in candidates:
        cfg_clean = {k: v for k, v in cfg.items() if k != "SPLIT_K"}
        try:
            result = verify_and_bench(
                kernel_fn=kernel_fn,
                reference_output=ref,
                config=cfg_clean,
                warmup=warmup,
                runs=runs,
                data_bytes=data_bytes,
                baseline_latency_us=baseline_latency,
                atol=1e-2,
                rtol=1e-2,
            )
            if result.correct and result.latency_us < best_latency:
                best_latency = result.latency_us
                best_config = cfg_clean
        except Exception:
            continue

    speedup = baseline_latency / best_latency if best_latency > 0 else None

    # Convert to codegen format
    codegen_config = {
        "nwarps": best_config.get("num_warps", 4),
        "rows_per_block": best_config.get("BLOCK_N", 64) // 64,  # normalize to rows
    }
    # Ensure rows_per_block is at least 1
    if codegen_config["rows_per_block"] < 1:
        codegen_config["rows_per_block"] = 1

    return codegen_config, baseline_latency, speedup


def cmd_gguf_optimize(args):
    """Parse GGUF, tune each unique shape, emit C header."""
    gguf_path = Path(args.gguf)
    if not gguf_path.exists():
        console.print(f"[red]GGUF file not found: {gguf_path}[/red]")
        sys.exit(1)

    # Check GPU (skip if --no-bench)
    no_bench = getattr(args, "no_bench", False)
    if not no_bench and not torch.cuda.is_available():
        console.print("[yellow]GPU not detected by PyTorch. Falling back to --no-bench mode.[/yellow]")
        console.print("[dim]To enable GPU benchmarking, ensure PyTorch is the ROCm build:[/dim]")
        console.print("[dim]  pip install torch --index-url https://download.pytorch.org/whl/rocm7.1/[/dim]")
        console.print("[dim]Or try: kernel-anvil autoforge model.gguf (uses hipcc directly)[/dim]")
        console.print()
        no_bench = True

    if no_bench:
        gpu_spec = GFX1100  # Default heuristics
        device = None
        console.print("[yellow]--no-bench: using heuristic configs (no GPU benchmarking)[/yellow]")
    else:
        gpu_spec = _get_gpu_spec()
        device = torch.device("cuda")

    # Parse GGUF
    console.print(f"\n[bold]Parsing {gguf_path.name}...[/bold]")
    profile = parse_gguf(str(gguf_path))

    # Print model summary
    console.print()
    print_model_summary(profile)
    console.print()

    # Collect unique 2D shapes to tune
    shapes = profile.unique_shapes
    if not shapes:
        console.print("[yellow]No 2D weight tensors found in model.[/yellow]")
        sys.exit(0)

    codegen_configs: dict[tuple[str, int, int], dict] = {}
    results_table = []
    total_t0 = time.monotonic()

    if no_bench:
        # Heuristic mode: assign configs based on shape analysis, no GPU needed
        console.print(f"[bold]Generating heuristic configs for {len(shapes)} shapes...[/bold]\n")
        for i, ((qt, n, k), count) in enumerate(sorted(shapes.items()), 1):
            blocks_per_row = k // 256 if k >= 256 else 1
            # Heuristic: fewer warps for small K (less sync overhead)
            if blocks_per_row < 64:  # small_k
                cfg = {"nwarps": 2, "rows_per_block": 2}
            else:
                cfg = {"nwarps": 4, "rows_per_block": 1}
            codegen_configs[(qt, n, k)] = cfg
            console.print(f"  [{i}/{len(shapes)}] {qt} ({n}, {k}) x{count}: "
                          f"nwarps={cfg['nwarps']} rows={cfg['rows_per_block']} (heuristic)")
            results_table.append((qt, n, k, count, cfg, 0, None, 0))
    else:
        console.print(f"[bold]Tuning {len(shapes)} unique GEMV workloads...[/bold]\n")

        for i, ((qt, n, k), count) in enumerate(sorted(shapes.items()), 1):
            console.print(
                f"  [{i}/{len(shapes)}] {qt} ({n}, {k}) x{count}...",
                end=" ",
            )
            t0 = time.monotonic()
            try:
                cfg, baseline_us, speedup = _tune_shape_cli(
                    N=n,
                    K=k,
                    device=device,
                    gpu_spec=gpu_spec,
                    max_configs=args.max_configs,
                    warmup=args.warmup,
                    runs=args.runs,
                )
                dt = time.monotonic() - t0
                speedup_str = f"{speedup:.2f}x" if speedup is not None else "-"
                console.print(
                    f"nwarps={cfg['nwarps']} rows={cfg['rows_per_block']} "
                    f"({speedup_str}, {dt:.1f}s)"
                )
                codegen_configs[(qt, n, k)] = cfg
                results_table.append((qt, n, k, count, cfg, baseline_us, speedup, dt))
            except Exception as e:
                console.print(f"[red]FAILED: {e}[/red]")
                codegen_configs[(qt, n, k)] = {"nwarps": 4, "rows_per_block": 1}
                results_table.append((qt, n, k, count, {"nwarps": 4, "rows_per_block": 1}, 0, None, 0))

    total_dt = time.monotonic() - total_t0

    # Print results summary table
    console.print()
    table = Table(title="Optimization Results")
    table.add_column("Quant", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("K", justify="right")
    table.add_column("Count", justify="right", style="dim")
    table.add_column("nwarps", justify="right")
    table.add_column("rows", justify="right")
    table.add_column("Baseline (us)", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Time", justify="right", style="dim")

    for qt, n, k, count, cfg, baseline_us, speedup, dt in results_table:
        speedup_str = f"[green]{speedup:.2f}x[/green]" if speedup is not None and speedup > 1.0 else (
            f"{speedup:.2f}x" if speedup is not None else "-"
        )
        table.add_row(
            qt,
            str(n),
            str(k),
            str(count),
            str(cfg["nwarps"]),
            str(cfg["rows_per_block"]),
            f"{baseline_us:.1f}" if baseline_us else "-",
            speedup_str,
            f"{dt:.1f}s",
        )

    console.print(table)
    console.print(f"\nTotal tuning time: {total_dt:.1f}s")

    # Generate runtime JSON config for llama.cpp auto-loading
    from kernel_anvil.codegen import generate_runtime_config

    json_config = generate_runtime_config(
        codegen_configs,
        gpu_name=gpu_spec.gfx,
        model_name=profile.name,
    )

    # Write to ~/.cache/smithy/<model_basename>.json for auto-loading
    # Must be ~/.cache/smithy/ to match llama.cpp's smithy-config.h lookup path
    model_basename = Path(args.gguf).stem  # e.g., "Qwen3-8B-Q4_K_M"
    cache_dir = Path.home() / ".cache" / "smithy"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{model_basename}.json"
    # Atomic write: write to temp file, then rename
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(cache_dir), suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(json_config)
        os.rename(tmp_path, str(cache_path))
    except Exception:
        os.unlink(tmp_path)
        raise
    console.print(f"\n[bold green]Config cached to {cache_path}[/bold green]")
    console.print("[dim]llama.cpp will auto-load this on next model load.[/dim]")

    # Also write C header if --output was explicitly set
    if args.output != "smithy-config.h":
        header = generate_config_header(
            codegen_configs,
            gpu_name=f"{gpu_spec.gfx} ({gpu_spec.name})",
            model_name=profile.name,
        )
        output_path = Path(args.output)
        output_path.write_text(header)
        console.print(f"[dim]C header also written to {output_path}[/dim]")


def cmd_autoforge(args):
    """Auto-generate optimized HIP kernels for a model."""
    from kernel_anvil.autoforge import autoforge

    try:
        nwarps = [int(x) for x in args.nwarps.split(",")]
        rpb = [int(x) for x in args.rpb.split(",")]
        if any(v <= 0 for v in nwarps + rpb):
            raise ValueError("All values must be positive")
    except ValueError as e:
        console.print(f"[red]Invalid nwarps/rpb values: {e}[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Autoforge: generating optimized kernels for {Path(args.gguf).name}[/bold]")
    console.print(f"Sweeping nwarps={nwarps} x rpb={rpb} = {len(nwarps)*len(rpb)} configs per shape\n")

    # Pass llama.cpp path if provided via CLI or env var
    llama_cpp_path = getattr(args, "llama_cpp_path", None)

    result = autoforge(
        model_path=args.gguf,
        arch=args.arch,
        nwarps_candidates=nwarps,
        rpb_candidates=rpb,
        llama_cpp_path=llama_cpp_path,
        verbose=True,
    )

    if result.shapes:
        console.print(f"\n[bold green]Done![/bold green]")
        console.print(f"  Config: {result.kernel_pack_path}")
        console.print(f"  Run: SMITHY_CONFIG={result.kernel_pack_path} llama-server -m {args.gguf} -ngl 999")


def cmd_llama_sweep(args):
    """Sweep actual llama.cpp MMVQ kernel configs via rocprofv3."""
    from kernel_anvil.llama_sweep import sweep_model, write_optimal_config

    nwarps = [int(x) for x in args.nwarps.split(",")]

    console.print(f"\n[bold]Sweeping llama.cpp MMVQ kernels for {Path(args.gguf).name}[/bold]")
    console.print(f"Candidate nwarps: {nwarps}")
    console.print(f"This runs llama-bench {len(nwarps)+1} times with rocprofv3 tracing.\n")

    result = sweep_model(
        model_path=args.gguf,
        llama_bench=args.llama_bench,
        nwarps_candidates=nwarps,
        verbose=True,
    )

    path = write_optimal_config(result)
    console.print(f"\n[bold green]Optimal config written to {path}[/bold green]")
    console.print(f"  Baseline: {result.baseline_tps:.2f} tok/s")
    console.print(f"  Optimized: {result.optimized_tps:.2f} tok/s")
    speedup = result.optimized_tps / result.baseline_tps if result.baseline_tps > 0 else 0
    console.print(f"  Speedup: {speedup:.2f}x")
    console.print(f"\n[dim]Run llama.cpp with: SMITHY_CONFIG={path} llama-server -m {args.gguf} -ngl 999[/dim]")


def cmd_compare(args):
    """Compare ROCm vs Vulkan backend performance."""
    from kernel_anvil.vulkan_sweep import compare_backends

    console.print(f"\n[bold]Backend Comparison: ROCm vs Vulkan[/bold]")
    console.print(f"Model: {Path(args.gguf).name}\n")

    results = compare_backends(
        model_path=args.gguf,
        rocm_bench=args.rocm_bench,
        vulkan_bench=args.vulkan_bench,
        verbose=True,
    )

    if "comparison" in results:
        rec = results["comparison"]["recommendation"]
        if rec == "vulkan":
            console.print(f"\n[bold green]Vulkan recommended for this workload.[/bold green]")
        elif rec == "rocm":
            console.print(f"\n[bold green]ROCm recommended for this workload.[/bold green]")


def main():
    parser = argparse.ArgumentParser(
        prog="kernel-anvil",
        description="Profile-guided GPU kernel optimizer for AMD",
    )
    sub = parser.add_subparsers(dest="command")

    # sweep
    p_sweep = sub.add_parser("sweep", help="End-to-end optimization sweep")
    p_sweep.add_argument("runner", help="Path to runner script")
    p_sweep.add_argument("--max-configs", type=int, default=20, help="Max configs to try (default: 20)")
    p_sweep.add_argument("--warmup", type=int, default=5, help="Warmup iterations (default: 5)")
    p_sweep.add_argument("--runs", type=int, default=10, help="Timed iterations (default: 10)")
    p_sweep.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance (default: 1e-2)")
    p_sweep.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance (default: 1e-2)")

    # profile
    p_profile = sub.add_parser("profile", help="Profile kernel and classify bottleneck")
    p_profile.add_argument("runner", help="Path to runner script")
    p_profile.add_argument("--warmup", type=int, default=5, help="Warmup iterations (default: 5)")
    p_profile.add_argument("--runs", type=int, default=20, help="Timed iterations (default: 20)")

    # gguf-optimize
    p_gguf = sub.add_parser("gguf-optimize", help="Parse GGUF, tune shapes, emit C header")
    p_gguf.add_argument("gguf", help="Path to GGUF model file")
    p_gguf.add_argument("--output", default="smithy-config.h", help="Output header path (default: smithy-config.h)")
    p_gguf.add_argument("--max-configs", type=int, default=15, help="Max configs per shape (default: 15)")
    p_gguf.add_argument("--warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    p_gguf.add_argument("--runs", type=int, default=5, help="Timed iterations (default: 5)")
    p_gguf.add_argument("--no-bench", action="store_true", help="Skip GPU benchmarking, use heuristic configs (works without GPU)")

    # autoforge: generate, compile, benchmark shape-specific kernels
    p_forge = sub.add_parser("autoforge", help="Auto-generate optimized HIP kernels for a model")
    p_forge.add_argument("gguf", help="Path to GGUF model file")
    p_forge.add_argument("--arch", help="GPU arch (auto-detected if omitted)")
    p_forge.add_argument("--nwarps", default="1,2,4,8", help="nwarps to sweep (default: 1,2,4,8)")
    p_forge.add_argument("--rpb", default="1,2,4", help="rows_per_block to sweep (default: 1,2,4)")
    p_forge.add_argument("--llama-cpp-path", help="Path to llama.cpp source (for headers). Also: LLAMA_CPP_PATH env var")

    # llama-sweep: sweep actual llama.cpp kernels via rocprofv3
    p_llama = sub.add_parser("llama-sweep", help="Sweep llama.cpp MMVQ kernel configs on actual hardware")
    p_llama.add_argument("gguf", help="Path to GGUF model file")
    p_llama.add_argument("--llama-bench", help="Path to llama-bench binary")
    p_llama.add_argument("--nwarps", default="1,2,4,8", help="Comma-separated nwarps to try (default: 1,2,4,8)")

    # compare: head-to-head ROCm vs Vulkan
    p_compare = sub.add_parser("compare-backends", help="Compare ROCm vs Vulkan decode performance")
    p_compare.add_argument("gguf", help="Path to GGUF model file")
    p_compare.add_argument("--rocm-bench", help="Path to ROCm llama-bench binary")
    p_compare.add_argument("--vulkan-bench", help="Path to Vulkan llama-bench binary")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "sweep":
        cmd_sweep(args)
    elif args.command == "profile":
        cmd_profile(args)
    elif args.command == "gguf-optimize":
        cmd_gguf_optimize(args)
    elif args.command == "llama-sweep":
        cmd_llama_sweep(args)
    elif args.command == "autoforge":
        cmd_autoforge(args)
    elif args.command == "compare-backends":
        cmd_compare(args)


if __name__ == "__main__":
    main()
