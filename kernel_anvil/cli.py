"""CLI for kernel-anvil -- profile-guided Triton kernel optimizer."""
import argparse
import importlib.util
import json
import os
import sys
import tempfile
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


def _profile_gguf_shapes(
    profile,
    *,
    no_bench: bool,
    gpu_spec,
    device,
    args,
    label: str,
    codegen_configs: dict,
    results_table: list,
    speedups: dict[tuple[str, int, int], float],
):
    """Tune every unique shape in ``profile`` and accumulate into the shared
    output dicts. Skips shapes already tuned by a previous model in the same
    invocation (target+draft sharing).

    On profiling failure for a shape, the slot is left ABSENT from
    ``codegen_configs`` (recorded in ``results_table`` as a FAIL row) so a
    subsequent draft model sharing that shape gets a fresh profiling
    attempt instead of being silently locked into the failure config."""
    shapes = profile.unique_shapes
    label_prefix = f"[{label}] " if label else ""

    if no_bench:
        console.print(f"[bold]{label_prefix}Generating heuristic configs for {len(shapes)} shapes...[/bold]\n")
        for i, ((qt, n, k), count) in enumerate(sorted(shapes.items()), 1):
            if (qt, n, k) in codegen_configs:
                console.print(f"  {label_prefix}[{i}/{len(shapes)}] {qt} ({n}, {k}) x{count}: [dim]reused[/dim]")
                continue
            blocks_per_row = k // 256 if k >= 256 else 1
            if blocks_per_row < 64:
                cfg = {"nwarps": 2, "rows_per_block": 2}
            else:
                cfg = {"nwarps": 4, "rows_per_block": 1}
            codegen_configs[(qt, n, k)] = cfg
            console.print(f"  {label_prefix}[{i}/{len(shapes)}] {qt} ({n}, {k}) x{count}: "
                          f"nwarps={cfg['nwarps']} rows={cfg['rows_per_block']} (heuristic)")
            results_table.append((label, qt, n, k, count, cfg, 0, None, 0))
        return

    console.print(f"[bold]{label_prefix}Tuning {len(shapes)} unique GEMV workloads...[/bold]\n")
    for i, ((qt, n, k), count) in enumerate(sorted(shapes.items()), 1):
        if (qt, n, k) in codegen_configs:
            console.print(f"  {label_prefix}[{i}/{len(shapes)}] {qt} ({n}, {k}) x{count}: [dim]reused[/dim]")
            continue
        console.print(
            f"  {label_prefix}[{i}/{len(shapes)}] {qt} ({n}, {k}) x{count}...",
            end=" ",
        )
        t0 = time.monotonic()
        try:
            cfg, baseline_us, speedup = _tune_shape_cli(
                N=n, K=k, device=device, gpu_spec=gpu_spec,
                max_configs=args.max_configs, warmup=args.warmup, runs=args.runs,
            )
            dt = time.monotonic() - t0
            speedup_str = f"{speedup:.2f}x" if speedup is not None else "-"
            console.print(
                f"nwarps={cfg['nwarps']} rows={cfg['rows_per_block']} "
                f"({speedup_str}, {dt:.1f}s)"
            )
            codegen_configs[(qt, n, k)] = cfg
            if speedup is not None:
                speedups[(qt, n, k)] = float(speedup)
            results_table.append((label, qt, n, k, count, cfg, baseline_us, speedup, dt))
        except Exception as e:
            console.print(f"[red]FAILED: {e}[/red]")
            # Don't poison the slot: leave (qt, n, k) absent from
            # codegen_configs so a subsequent draft model with the same shape
            # can still try profiling it. The results_table tracks the
            # failure for the user-visible summary.
            results_table.append((label, qt, n, k, count, None, 0, None, 0))


def cmd_gguf_optimize(args):
    """Parse GGUF (and optional draft GGUFs), tune each unique GEMV shape,
    and write a runtime JSON config to ``~/.cache/smithy/<stem>.json``.

    With ``--draft <gguf>`` (repeatable), profiles target and draft(s)
    together and merges into a single config keyed under the target stem.
    Bucket-cell collisions resolve to the higher profiled speedup so the
    better-performing config survives.

    A C header is also emitted when ``--output`` is explicitly set to a
    non-default path."""
    gguf_path = Path(args.gguf)
    if not gguf_path.exists():
        console.print(f"[red]GGUF file not found: {gguf_path}[/red]")
        sys.exit(1)

    draft_paths: list[Path] = []
    for raw in getattr(args, "draft", None) or []:
        dp = Path(raw)
        if not dp.exists():
            console.print(f"[red]Draft GGUF not found: {dp}[/red]")
            sys.exit(1)
        draft_paths.append(dp)

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

    # Parse target (and optional drafts)
    console.print(f"\n[bold]Parsing target {gguf_path.name}...[/bold]")
    target_profile = parse_gguf(str(gguf_path))
    console.print()
    print_model_summary(target_profile)
    console.print()

    draft_profiles = []
    for dp in draft_paths:
        console.print(f"\n[bold]Parsing draft {dp.name}...[/bold]")
        dprof = parse_gguf(str(dp))
        console.print()
        print_model_summary(dprof)
        console.print()
        draft_profiles.append((dp, dprof))

    if not target_profile.unique_shapes and not any(p.unique_shapes for _, p in draft_profiles):
        console.print("[yellow]No 2D weight tensors found in any model.[/yellow]")
        sys.exit(0)

    codegen_configs: dict[tuple[str, int, int], dict] = {}
    speedups: dict[tuple[str, int, int], float] = {}
    results_table: list = []
    total_t0 = time.monotonic()

    _profile_gguf_shapes(
        target_profile,
        no_bench=no_bench, gpu_spec=gpu_spec, device=device, args=args,
        label="target" if draft_profiles else "",
        codegen_configs=codegen_configs, results_table=results_table, speedups=speedups,
    )
    for idx, (_, dprof) in enumerate(draft_profiles, start=1):
        label = "draft" if len(draft_profiles) == 1 else f"draft{idx}"
        _profile_gguf_shapes(
            dprof,
            no_bench=no_bench, gpu_spec=gpu_spec, device=device, args=args,
            label=label,
            codegen_configs=codegen_configs, results_table=results_table, speedups=speedups,
        )

    total_dt = time.monotonic() - total_t0

    # Print results summary table
    console.print()
    table = Table(title="Optimization Results")
    if draft_profiles:
        table.add_column("Source", style="magenta")
    table.add_column("Quant", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("K", justify="right")
    table.add_column("Count", justify="right", style="dim")
    table.add_column("nwarps", justify="right")
    table.add_column("rows", justify="right")
    table.add_column("Baseline (us)", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Time", justify="right", style="dim")

    for label, qt, n, k, count, cfg, baseline_us, speedup, dt in results_table:
        speedup_str = f"[green]{speedup:.2f}x[/green]" if speedup is not None and speedup > 1.0 else (
            f"{speedup:.2f}x" if speedup is not None else "-"
        )
        nwarps_str = str(cfg["nwarps"]) if cfg else "[red]FAIL[/red]"
        rpb_str = str(cfg["rows_per_block"]) if cfg else "[red]FAIL[/red]"
        row = [
            qt, str(n), str(k), str(count),
            nwarps_str, rpb_str,
            f"{baseline_us:.1f}" if baseline_us else "-",
            speedup_str, f"{dt:.1f}s",
        ]
        if draft_profiles:
            row.insert(0, label or "target")
        table.add_row(*row)

    console.print(table)
    console.print(f"\nTotal tuning time: {total_dt:.1f}s")

    # Generate runtime JSON config for llama.cpp auto-loading
    from kernel_anvil.codegen import generate_runtime_config

    if draft_profiles:
        names = [target_profile.name] + [p.name for _, p in draft_profiles]
        merged_model_name = "+".join(names)
    else:
        merged_model_name = target_profile.name

    json_config = generate_runtime_config(
        codegen_configs,
        gpu_name=gpu_spec.gfx,
        model_name=merged_model_name,
        priorities=speedups if draft_profiles else None,
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
    finally:
        # Whether rename succeeded, raised OSError, or was aborted by an
        # unrelated exception (KeyboardInterrupt, MemoryError, etc.), make
        # sure we don't leak tempfiles in the cache dir.
        if Path(tmp_path).exists():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    console.print(f"\n[bold green]Config cached to {cache_path}[/bold green]")
    if draft_profiles:
        # llama.cpp's llama-server only accepts ONE -md flag at runtime, so
        # the hint emits a single draft. The merged config still covers any
        # additional draft GGUFs the user passed via --draft, which is
        # useful when iterating across draft candidates without re-tuning.
        primary_draft = draft_profiles[0][0]
        console.print(
            f"[dim]Run: SMITHY_CONFIG={cache_path} llama-server "
            f"-m {args.gguf} -md {primary_draft} -ngl 999[/dim]"
        )
        if len(draft_profiles) > 1:
            extras = ", ".join(str(dp) for dp, _ in draft_profiles[1:])
            console.print(
                f"[dim]      (merged config also covers: {extras})[/dim]"
            )
    else:
        console.print(
            f"[dim]Run: SMITHY_MODEL={args.gguf} llama-server -m {args.gguf} -ngl 999[/dim]"
        )

    # Also write C header if --output was explicitly set
    if args.output != "smithy-config.h":
        header = generate_config_header(
            codegen_configs,
            gpu_name=f"{gpu_spec.gfx} ({gpu_spec.name})",
            model_name=merged_model_name,
        )
        output_path = Path(args.output)
        output_path.write_text(header)
        console.print(f"[dim]C header also written to {output_path}[/dim]")


def cmd_merge_configs(args):
    """Merge multiple kernel-anvil JSON configs into one.

    Useful for speculative-decoding setups: profile each model separately
    (target + draft) with ``gguf-optimize``, then point ``SMITHY_CONFIG`` at
    the merged output. First-listed input wins on bucket-cell collisions.
    """
    import json
    from kernel_anvil.codegen import merge_runtime_configs

    payloads = []
    for raw in args.inputs:
        p = Path(raw)
        if not p.exists():
            console.print(f"[red]Config not found: {p}[/red]")
            sys.exit(1)
        try:
            with open(p) as f:
                payloads.append(json.load(f))
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON in {p}: {e}[/red]")
            sys.exit(1)

    try:
        merged = merge_runtime_configs(
            payloads,
            gpu_name=args.gpu,
            model_name=args.model,
        )
    except (TypeError, ValueError, AttributeError) as e:
        console.print(f"[red]Invalid config payload: {e}[/red]")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: write to temp file in the same dir, then rename. Mirrors
    # the cmd_gguf_optimize pattern; prevents truncated output if two
    # merge-configs runs target the same path or the process is killed.
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(out_path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(json.dumps(merged, indent=2))
        os.rename(tmp_path, str(out_path))
    finally:
        # Always clean up the tempfile if it's still there (covers OSError,
        # KeyboardInterrupt, MemoryError, anything that aborted the rename).
        if Path(tmp_path).exists():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    n_types = len(merged.get("configs") or {})
    n_cells = sum(len(v) for v in (merged.get("configs") or {}).values())
    console.print(f"[bold green]Merged {len(payloads)} configs -> {out_path}[/bold green]")
    console.print(f"[dim]{n_types} quant types, {n_cells} bucket cells[/dim]")
    console.print(f"[dim]Run: SMITHY_CONFIG={out_path} llama-server -m target.gguf -md draft.gguf -ngl 999[/dim]")


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


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically write text to ``path`` via a tempfile + os.rename.

    Mirrors the cmd_gguf_optimize pattern so train-optimize gets the same
    crash-safety guarantees (no truncated JSON on Ctrl-C, no leftover
    `.json.tmp` files in the cache dir).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(text)
        os.rename(tmp_path, str(path))
    finally:
        if Path(tmp_path).exists():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _resolve_train_runner(op: str, mud_puppy_path: str | None):
    """Look up a mud-puppy training runner module for the given op.

    The runner contract lives in `mud_puppy.anvil_runner.<op>_runner` (e.g.
    `mxfp4_fwd_runner`). When ``mud_puppy_path`` is provided we prepend it
    to sys.path so the import resolves against a checkout that isn't
    pip-installed.

    Returns the module object on success, None when the runner is not
    importable. Caller decides whether to skip (dry-run) or hard-fail.
    """
    if mud_puppy_path:
        mp_path = str(Path(mud_puppy_path).resolve())
        if mp_path not in sys.path:
            sys.path.insert(0, mp_path)
    try:
        import importlib

        module_name = f"mud_puppy.anvil_runner.{op}_runner"
        return importlib.import_module(module_name)
    except Exception:
        return None


def _train_dry_run_config(M: int, N: int, K: int) -> dict:
    """Synthetic 'best config' used when no GPU/runner is available.

    The default mirrors the empirical winner from the headroom benchmark
    (BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_warps=8,
    num_stages=4) so dry-run JSON skeletons are immediately useful as a
    starting point if a real sweep can't be done."""
    return {
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "BLOCK_K": 32,
        "GROUP_M": 8,
        "num_warps": 8,
        "num_stages": 4,
        "speedup_vs_baseline": 1.0,
        "profiled_us": 0.0,
    }


def _tune_train_shape(
    *,
    runner_module,
    op: str,
    M: int,
    N: int,
    K: int,
    gpu_spec,
    max_configs: int,
    warmup: int,
    runs: int,
) -> tuple[dict, float, float | None]:
    """Sweep a single (op, M, N, K) shape via a mud-puppy runner module.

    The runner is expected to expose:
        - `make_inputs(M, N, K) -> inputs` (preferred) OR `setup() -> inputs`
        - `reference(inputs) -> tensor or tuple`
        - `run(inputs, **config) -> tensor or tuple`
        - optional `BASELINE_CONFIG` and `DATA_BYTES`
        - optional `OUTPUT_INDEX` for which slot to allclose-check (default 0)

    Returns ``(best_payload, baseline_us, speedup_or_None)`` where
    `best_payload` is a dict suitable for `train_codegen.generate_train_runtime_config`.
    """
    from kernel_anvil.train_param_space import generate_train_configs

    # Build inputs. Prefer make_inputs(M, N, K) if exposed; fall back to
    # setup() with no args (legacy contract).
    if hasattr(runner_module, "make_inputs"):
        inputs = runner_module.make_inputs(M, N, K)
    else:
        inputs = runner_module.setup()

    ref = runner_module.reference(inputs)
    output_index = int(getattr(runner_module, "OUTPUT_INDEX", 0))
    baseline_config = dict(getattr(runner_module, "BASELINE_CONFIG", {}) or {})
    data_bytes = getattr(runner_module, "DATA_BYTES", None)

    def kernel_fn(cfg):
        return runner_module.run(inputs, **cfg)

    # Baseline benchmark (if BASELINE_CONFIG present, otherwise skip).
    baseline_latency = None
    if baseline_config:
        try:
            baseline_result = verify_and_bench(
                kernel_fn=kernel_fn,
                reference_output=ref,
                config=baseline_config,
                warmup=warmup,
                runs=runs,
                data_bytes=data_bytes,
                output_index=output_index,
            )
            baseline_latency = baseline_result.latency_us
        except Exception:
            baseline_latency = None

    # Generate candidate configs and benchmark each.
    candidates = generate_train_configs(gpu=gpu_spec, max_configs=max_configs)
    best_cfg = baseline_config or candidates[0] if candidates else baseline_config
    best_latency = baseline_latency if baseline_latency is not None else float("inf")

    for cfg in candidates:
        try:
            result = verify_and_bench(
                kernel_fn=kernel_fn,
                reference_output=ref,
                config=cfg,
                warmup=warmup,
                runs=runs,
                data_bytes=data_bytes,
                baseline_latency_us=baseline_latency,
                atol=1e-2,
                rtol=1e-2,
                output_index=output_index,
            )
            if result.correct and result.latency_us < best_latency:
                best_latency = result.latency_us
                best_cfg = cfg
        except Exception:
            continue

    if best_cfg is None:
        # Everything failed -- emit a dry-run payload so the JSON layer
        # still has a sensible default for this cell.
        return _train_dry_run_config(M, N, K), baseline_latency or 0.0, None

    speedup = None
    if baseline_latency is not None and best_latency > 0:
        speedup = baseline_latency / best_latency

    payload = {
        "BLOCK_M": int(best_cfg.get("BLOCK_M", 128)),
        "BLOCK_N": int(best_cfg.get("BLOCK_N", 64)),
        "BLOCK_K": int(best_cfg.get("BLOCK_K", 32)),
        "GROUP_M": int(best_cfg.get("GROUP_M", 8)),
        "num_warps": int(best_cfg.get("num_warps", 8)),
        "num_stages": int(best_cfg.get("num_stages", 4)),
    }
    if speedup is not None:
        payload["speedup_vs_baseline"] = float(speedup)
    if best_latency != float("inf"):
        payload["profiled_us"] = float(best_latency)
    return payload, float(baseline_latency or 0.0), speedup


def cmd_train_optimize(args):
    """Profile training-time GEMM shapes for a model and emit anvil-train JSON.

    The training-side counterpart of `cmd_gguf_optimize`. Walks the HF
    model config, enumerates the unique (op, M, N, K) shapes for the given
    (batch, seq), runs a sweep against the mud-puppy runner for each, and
    writes a v1 JSON to `~/.cache/anvil-train/<gpu>/<model>-<quant>-bN-sM.json`.

    When mud-puppy isn't importable (no ``--mud-puppy-path`` and no
    installed package), falls back to a dry-run mode that emits a JSON
    skeleton with placeholder configs -- useful for testing the JSON layer
    on a CPU-only box.
    """
    from kernel_anvil.train_codegen import generate_train_runtime_config
    from kernel_anvil.train_shapes import extract_shapes, model_basename

    # Resolve quant -> ops mapping
    quant = args.quant
    if quant not in ("mxfp4", "int4"):
        console.print(f"[red]Unknown quant: {quant} (expected mxfp4 or int4)[/red]")
        sys.exit(1)

    if args.ops:
        ops = [s.strip() for s in args.ops.split(",") if s.strip()]
    else:
        ops = [f"{quant}_fwd", f"{quant}_grad_input"]

    dry_run = bool(getattr(args, "dry_run", False))

    # Detect GPU for cache-path stamping (always succeeds; falls back to GFX1100).
    if dry_run:
        gpu_spec = GFX1100
        device = None
        console.print("[yellow]--dry-run: emitting JSON skeleton, no GPU benchmarking[/yellow]")
    else:
        if not torch.cuda.is_available():
            console.print(
                "[yellow]GPU not detected -- falling back to --dry-run mode."
                " Install a ROCm PyTorch build to benchmark training kernels.[/yellow]"
            )
            dry_run = True
            gpu_spec = GFX1100
            device = None
        else:
            gpu_spec = _get_gpu_spec()
            device = torch.device("cuda")

    # Pull shapes
    console.print(f"\n[bold]Extracting training shapes for {args.model}...[/bold]")
    try:
        shapes = extract_shapes(args.model, batch=args.batch, seq=args.seq, ops=ops)
    except Exception as exc:
        console.print(f"[red]Failed to extract shapes: {exc}[/red]")
        sys.exit(1)

    if not shapes:
        console.print("[yellow]No shapes extracted -- nothing to do.[/yellow]")
        sys.exit(0)

    console.print(f"[dim]{len(shapes)} unique (op, M, N, K) tuples[/dim]")

    # Resolve runners (one per op family)
    runners: dict[str, object] = {}
    if not dry_run:
        for op in ops:
            mod = _resolve_train_runner(op, args.mud_puppy_path)
            if mod is None:
                console.print(
                    f"[yellow]Runner for {op!r} not available (no mud-puppy on path);"
                    f" falling back to dry-run for this op.[/yellow]"
                )
            runners[op] = mod
        if all(r is None for r in runners.values()):
            console.print(
                "[yellow]No runners resolved -- switching entire run to --dry-run mode.[/yellow]"
            )
            dry_run = True

    # Sweep loop
    configs: dict[tuple[str, int, int, int], dict] = {}
    speedups: dict[tuple[str, int, int, int], float] = {}
    results_table_rows: list = []
    total_t0 = time.monotonic()

    for i, (op, M, N, K) in enumerate(shapes, 1):
        runner = runners.get(op) if not dry_run else None
        t0 = time.monotonic()
        if runner is None or dry_run:
            payload = _train_dry_run_config(M, N, K)
            speedup = None
            baseline_us = 0.0
        else:
            try:
                payload, baseline_us, speedup = _tune_train_shape(
                    runner_module=runner,
                    op=op,
                    M=M,
                    N=N,
                    K=K,
                    gpu_spec=gpu_spec,
                    max_configs=args.max_configs,
                    warmup=args.warmup,
                    runs=args.runs,
                )
            except Exception as exc:
                console.print(f"  [{i}/{len(shapes)}] {op} ({M},{N},{K}): [red]FAILED[/red] {exc}")
                payload = _train_dry_run_config(M, N, K)
                speedup = None
                baseline_us = 0.0
        configs[(op, M, N, K)] = payload
        if speedup is not None:
            speedups[(op, M, N, K)] = float(speedup)
        dt = time.monotonic() - t0
        results_table_rows.append((op, M, N, K, payload, baseline_us, speedup, dt))
        if speedup is not None:
            console.print(
                f"  [{i}/{len(shapes)}] {op} ({M},{N},{K}): "
                f"BLOCK_M={payload['BLOCK_M']} BLOCK_N={payload['BLOCK_N']} "
                f"BLOCK_K={payload['BLOCK_K']} ({speedup:.2f}x, {dt:.1f}s)"
            )
        else:
            tag = "dry-run" if dry_run or runner is None else "no-baseline"
            console.print(
                f"  [{i}/{len(shapes)}] {op} ({M},{N},{K}): "
                f"BLOCK_M={payload['BLOCK_M']} BLOCK_N={payload['BLOCK_N']} "
                f"BLOCK_K={payload['BLOCK_K']} ({tag})"
            )

    total_dt = time.monotonic() - total_t0

    # Print summary table
    table = Table(title="Train-Optimize Results")
    table.add_column("Op", style="cyan")
    table.add_column("M", justify="right")
    table.add_column("N", justify="right")
    table.add_column("K", justify="right")
    table.add_column("BLOCK_M/N/K", justify="right")
    table.add_column("warps", justify="right")
    table.add_column("stages", justify="right")
    table.add_column("Baseline (us)", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Time", justify="right", style="dim")

    for op, M, N, K, payload, baseline_us, speedup, dt in results_table_rows:
        speedup_str = (
            f"[green]{speedup:.2f}x[/green]" if speedup is not None and speedup > 1.0
            else (f"{speedup:.2f}x" if speedup is not None else "-")
        )
        tile_str = f"{payload['BLOCK_M']}/{payload['BLOCK_N']}/{payload['BLOCK_K']}"
        table.add_row(
            op, str(M), str(N), str(K),
            tile_str,
            str(payload["num_warps"]),
            str(payload["num_stages"]),
            f"{baseline_us:.1f}" if baseline_us else "-",
            speedup_str,
            f"{dt:.1f}s",
        )

    console.print()
    console.print(table)
    console.print(f"\nTotal tuning time: {total_dt:.1f}s")

    # Build the JSON payload
    model_id = args.model
    basename = model_basename(model_id)
    rocm_version = _detect_rocm_version()
    torch_version = getattr(torch, "__version__", "")
    triton_version = _detect_triton_version()

    json_text = generate_train_runtime_config(
        configs,
        gpu=gpu_spec.gfx,
        model=basename,
        batch=args.batch,
        seq=args.seq,
        rocm_version=rocm_version,
        torch_version=torch_version,
        triton_version=triton_version,
        kernel_hash="",  # populated by mud-puppy on the loader side
        priorities={k: int(v * 1000) for k, v in speedups.items()} or None,
    )

    # Resolve output path
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        out_path = (
            Path.home() / ".cache" / "anvil-train" / gpu_spec.gfx
            / f"{basename}-{quant}-b{args.batch}s{args.seq}.json"
        )
    _atomic_write_text(out_path, json_text)

    console.print(f"\n[bold green]anvil-train config written to {out_path}[/bold green]")
    if dry_run:
        console.print(
            "[dim]Dry-run JSON. Re-run with mud-puppy on the path and a GPU"
            " to populate real configs.[/dim]"
        )


def _detect_rocm_version() -> str:
    """Best-effort ROCm version string. Empty when unavailable."""
    try:
        info_file = Path("/opt/rocm/.info/version")
        if info_file.exists():
            return info_file.read_text().strip()
    except OSError:
        pass
    return ""


def _detect_triton_version() -> str:
    try:
        import triton  # type: ignore

        return getattr(triton, "__version__", "")
    except Exception:
        return ""


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
    p_gguf.add_argument(
        "--draft",
        action="append",
        metavar="GGUF",
        help="Optional draft model GGUF (for speculative decoding). May be passed multiple times. "
             "Profiles target + draft together and writes a single merged config keyed under the target stem.",
    )

    # merge-configs: combine multiple cached JSON configs into one
    p_merge = sub.add_parser(
        "merge-configs",
        help="Merge multiple kernel-anvil JSON configs into one (e.g. for speculative decoding)",
    )
    p_merge.add_argument("inputs", nargs="+", help="Input config JSON files (priority = argument order)")
    p_merge.add_argument("-o", "--output", required=True, help="Output merged config path")
    p_merge.add_argument("--gpu", help="Override gpu field in merged output")
    p_merge.add_argument("--model", help="Override model field in merged output")

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

    # train-optimize: profile training-side GEMM shapes for mud-puppy
    p_train = sub.add_parser(
        "train-optimize",
        help="Tune training-time GEMM kernels (mxfp4/int4 fwd + grad-input)"
             " against an HF model + (batch, seq) shape -- emits anvil-train v1 JSON.",
    )
    p_train.add_argument("model", help="HF model id (e.g. Qwen/Qwen3-8B) or local path")
    p_train.add_argument(
        "--quant",
        choices=("mxfp4", "int4"),
        required=True,
        help="Quantization scheme to tune (selects op set unless --ops is given)",
    )
    p_train.add_argument("--batch", type=int, required=True, help="Per-device batch size")
    p_train.add_argument("--seq", type=int, required=True, help="Training sequence length")
    p_train.add_argument(
        "--ops",
        default="",
        help="Comma-separated op list. Default: <quant>_fwd,<quant>_grad_input",
    )
    p_train.add_argument(
        "--mud-puppy-path",
        default=None,
        help="Path to a mud-puppy checkout (prepended to sys.path so"
             " mud_puppy.anvil_runner.<op>_runner imports resolve).",
    )
    p_train.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Default: ~/.cache/anvil-train/<gpu>/<model>-<quant>-b<B>s<S>.json",
    )
    p_train.add_argument(
        "--max-configs", type=int, default=30,
        help="Cap on Triton configs per shape (default: 30).",
    )
    p_train.add_argument(
        "--warmup", type=int, default=3,
        help="Warmup iterations per benchmark (default: 3).",
    )
    p_train.add_argument(
        "--runs", type=int, default=5,
        help="Timed iterations per benchmark (default: 5).",
    )
    p_train.add_argument(
        "--dry-run", action="store_true",
        help="Skip GPU benchmarking; emit a JSON skeleton with placeholder"
             " configs. Useful for testing the JSON layer without a GPU.",
    )

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
    elif args.command == "merge-configs":
        cmd_merge_configs(args)
    elif args.command == "train-optimize":
        cmd_train_optimize(args)


if __name__ == "__main__":
    main()
