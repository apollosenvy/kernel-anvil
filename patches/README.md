# llama.cpp Patch for kernel-anvil Runtime Config

This patch adds runtime shape-specific MMVQ kernel configuration to llama.cpp.
When kernel-anvil generates optimized configs for your model + GPU, this patch
lets llama.cpp load them at startup.

## Patch Files

| File | Target | Required | Purpose |
|------|--------|----------|---------|
| `smithy-config.h` | `ggml/src/ggml-cuda/` | Yes | Config loader header (3-tier JSON lookup) |
| `mmvq-smithy.patch` | `ggml/src/ggml-cuda/mmvq.cu` | Yes | Hooks `smithy_lookup()` into MMVQ dispatch |
| `arg-h-smithy-router.patch` | `common/arg.h` | No | Registers `smithy-config` pseudo-env for models.ini |
| `arg-cpp-smithy-router.patch` | `common/arg.cpp` | No | Accepts `smithy-config` as a preset-only INI option |
| `server-models-cpp-smithy-router.patch` | `tools/server/server-models.cpp` | No | Injects `SMITHY_CONFIG` env at child spawn |

The three `*-router.patch` files enable per-model smithy config in llama.cpp's
router mode (multiple models loaded concurrently). They are applied
automatically by `apply.sh` when the llama.cpp source supports router mode;
on older trees they are silently skipped and the single-model `SMITHY_CONFIG`
env var still works.

## Requirements

`smithy-config.h` requires **C++17** (uses `std::atomic`, `std::shared_mutex`,
`[[maybe_unused]]`, and capture lambdas). llama.cpp's CMake already targets
C++17 by default, so no extra flags are needed for the upstream tree.

Your llama.cpp must be recent enough that `calc_rows_per_block` takes a
`bool small_k` parameter (mid-2026 upstream, including the ROCm/TheRock
fork). On older trees the patch's context will not match and `apply.sh`
will tell you to update.

Two variants of the patch ship, same hook, different context:

| File | Applies to |
|------|------------|
| `mmvq-smithy.patch` | ggml-org master from 2026-08-18 (commit `25ae3a9b3`, the GB10 MMVQ table) onward |
| `mmvq-smithy-pre-gb10.patch` | trees before that commit, including the ROCm/TheRock fork as of 2026-09 |

`apply.sh` tries them in that order and reports which one it used. If you
apply by hand, either file's hunks are the manual steps below.

## Quick Apply

```bash
cd kernel-anvil/patches
./apply.sh /path/to/llama.cpp
```

## Manual Apply

If the patch doesn't apply cleanly (llama.cpp version mismatch), make these changes by hand:

### 1. Copy `smithy-config.h`

Copy `patches/smithy-config.h` to `ggml/src/ggml-cuda/smithy-config.h` in your llama.cpp tree.

### 2. Edit `ggml/src/ggml-cuda/mmvq.cu`

**Add the include** (top of file, after other includes):
```cpp
#include "smithy-config.h"
```

**In `calc_rows_per_block`**, add before the final `return 1;`.

> **Careful about the neighboring function.** `calc_nwarps`, directly above,
> ends with an identical `return 1; }` tail. Make sure you are editing the
> function with this exact signature:
> `static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1)`
> If `small_k` is not a parameter of the function you are editing, you are in
> the wrong function and the build will fail with
> `use of undeclared identifier 'small_k'` (see issue #13).

```cpp
    // kernel-anvil (smithy): RDNA tables mirror GENERIC's small_k behavior.
    // Upstream's should_use_small_k blanket-disables small_k on RDNA, so on
    // AMD this branch is only reachable when a profiled smithy config
    // re-enabled the path for a shape measured faster with it. Host and
    // device share this constexpr, so launch params and kernel agree.
    if (small_k && ncols_dst == 1) {
        return nwarps;
    }
```

**In the `should_use_small_k` lambda** (inside `mul_mat_vec_q_switch_ncols_dst`),
add before `return use;`:
```cpp
        // kernel-anvil (smithy) per-shape override: a profiled config can
        // re-enable small_k where the blanket rules above (notably the
        // RDNA disable) leave measured speedup on the table for this GPU +
        // model's specific (type, nrows, ncols). Absent config = no-op.
        // Only forces the path ON, never off, matching the original patch.
        if (!use && c_ncols_dst == 1) {
            const smithy_shape_config scfg = smithy_lookup(type, nrows_x, ncols_x);
            if (scfg.rows_per_block > 1) {
                use = true;
            }
        }
```

### 3. Rebuild

```bash
cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

## How It Works

The patch is ~15 lines of actual code change:

1. `smithy-config.h` loads a JSON config file at first kernel dispatch
2. For decode (batch=1), it checks if kernel-anvil profiled this shape
3. If the config says `rows_per_block > 1`, it triggers the `small_k` kernel
   variant which processes multiple rows per block (`nwarps`) instead of 1
4. This cuts the number of kernel launches and improves occupancy

Without a config file, behavior is identical to stock llama.cpp.

## Config Loading

Checked in order:
1. `SMITHY_CONFIG` environment variable (explicit path to config JSON)
2. `~/.cache/smithy/<model_stem>.json` (model-specific, auto-generated by kernel-anvil)
3. `~/.cache/smithy/default.json` (fallback)

For step 2, the model stem is derived from the GGUF filename (e.g., `Qwen3-8B-Q4_K_M.gguf`
becomes `Qwen3-8B-Q4_K_M`). The model path is resolved from:
- `smithy_set_model()` C API (call from your model loading code), or
- `SMITHY_MODEL` environment variable (set to the GGUF file path)

Example usage:
```bash
# Automatic: kernel-anvil writes to ~/.cache/smithy/<model>.json, SMITHY_MODEL tells llama.cpp which one
kernel-anvil gguf-optimize model.gguf
SMITHY_MODEL=/path/to/model.gguf llama-server -m /path/to/model.gguf -ngl 999

# Explicit: point directly at a config file
SMITHY_CONFIG=~/.cache/smithy/Qwen3-8B-Q4_K_M.json llama-server -m model.gguf -ngl 999
```

Generate configs with:
```bash
kernel-anvil gguf-optimize model.gguf
# or for full benchmarking:
kernel-anvil autoforge model.gguf --llama-cpp-path /path/to/llama.cpp
```

## Router Mode (Per-Model Configs)

When running `llama-server` in router mode (multiple models loaded
concurrently), the `SMITHY_CONFIG` env var is process-wide and cannot vary
per model. The three `*-router.patch` files add a `smithy-config` key to
`models.ini` so each model section can point to its own config JSON:

```ini
[unsloth/Qwen3.6-27B-MTP-Vulkan-GGUF:Q8_K_XL]
smithy-config = ~/.cache/smithy/Qwen3.6-27B-MTP-Vulkan-GGUF.json

[satgeze/Hy3-1M-GGUF-tuned:Q4_K_M]
smithy-config = /models/smithy-cache/Hy3-1M-GGUF-tuned.json
```

### How it works

1. `arg-h-smithy-router.patch` + `arg-cpp-smithy-router.patch` register
   `smithy-config` as a preset-only INI option (like `load-on-startup`).
   This passes the INI parser's key validation without adding a CLI arg.
2. `server-models-cpp-smithy-router.patch` reads the value at child spawn
   and injects it as `SMITHY_CONFIG=<path>` into the child process
   environment.
3. The child's existing `smithy-config.h` picks up `SMITHY_CONFIG` as
   priority 1 in its 3-tier lookup — no changes to the loader needed.

Models without a `smithy-config` key fall back to the default lookup chain
(per-model-stem JSON, then `default.json`).

### Manual Apply (if router patches don't apply via `apply.sh`)

**`common/arg.h`** — add after the existing `COMMON_ARG_PRESET_*` defines:
```cpp
#define COMMON_ARG_PRESET_SMITHY_CONFIG  "__PRESET_SMITHY_CONFIG"
```

**`common/arg.cpp`** — in `common_params_add_preset_options()`, add after the
`stop-timeout` entry:
```cpp
args.push_back(common_arg(
    {"smithy-config"}, "PATH",
    "path to smithy per-model kernel config JSON (router mode: injected as SMITHY_CONFIG env for the child process)",
    [](common_params &, const std::string &) { /* unused */ }
).set_env(COMMON_ARG_PRESET_SMITHY_CONFIG).set_preset_only());
```

**`tools/server/server-models.cpp`** — in `server_models::load()`, after
`child_env = base_env;` and the `LLAMA_SERVER_ROUTER_PORT` push:
```cpp
// Inject per-model smithy config as SMITHY_CONFIG env var for the child
std::string smithy_path;
if (inst.meta.preset.get_option(COMMON_ARG_PRESET_SMITHY_CONFIG, smithy_path) && !smithy_path.empty()) {
    child_env.push_back("SMITHY_CONFIG=" + smithy_path);
}
```
