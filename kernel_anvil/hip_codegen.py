"""Generate shape-specialized HIP MMVQ kernels for llama.cpp.

Instead of tuning configs for the generic MMVQ kernel, this generates
purpose-built HIP kernels for specific (quant_type, N, K) shapes.
The generated kernels have:
- Hardcoded N, K dimensions (enables compiler optimizations)
- Optimal nwarps and rows_per_block for the specific shape
- Unrolled inner loops for known K
- Minimal cross-warp reduction (only as many warps as needed)

The output is a .hip file that hipcc compiles to a .so, which llama.cpp
can load via dlopen.

Two codegen paths:
1. Generic (preferred): #includes llama.cpp headers for vec_dot functions.
   Works for ALL quantization types. Requires llama.cpp source tree.
2. Fallback: Standalone Q4_K kernel with inlined dequant logic.
   No external dependencies but only supports Q4_K.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class KernelSpec:
    """Specification for a generated kernel."""
    quant_type: str  # "Q4_K", "Q6_K", etc.
    N: int           # output rows
    K: int           # input columns
    nwarps: int      # optimal warps per block
    rows_per_block: int  # output rows per block
    name: str = ""   # generated kernel function name

    def __post_init__(self):
        if not self.name:
            qt = self.quant_type.lower().replace("_", "")
            self.name = f"mmvq_{qt}_n{self.N}_k{self.K}_w{self.nwarps}_r{self.rows_per_block}"


@dataclass(frozen=True)
class QuantTypeInfo:
    """Quantization type metadata for kernel generation.

    These values come from llama.cpp's ggml-common.h and vecdotq.cuh.
    They define how the MMVQ inner loop iterates over quantized blocks.

    Attributes:
        name: Type name matching llama.cpp convention ("Q4_K", "Q6_K", etc.)
        ggml_type: Enum value from ggml_type in ggml.h.
        qk: Values per quantization block (QK_K=256 for K-quants, 32 for basic).
        qi: Number of int32s consumed per vec_dot call. Controls thread mapping.
        vdr: Vector dot results per call (VDR_*_Q8_1_MMVQ). Affects parallelism.
        block_size: sizeof(block_*) in bytes. Used for bandwidth calculations.
        vec_dot_func: Name of the vec_dot function in vecdotq.cuh.
    """
    name: str
    ggml_type: int
    qk: int
    qi: int
    vdr: int
    block_size: int
    vec_dot_func: str


# All MMVQ-capable quantization types.
# qi values computed from: qi = qk / (4 * qr), where qr comes from ggml-common.h.
# block_size values from sizeof(block_*) in ggml-common.h.
# vdr values from VDR_*_Q8_1_MMVQ in vecdotq.cuh.
QUANT_TYPES: dict[str, QuantTypeInfo] = {
    # Basic quants (qk=32)
    # qi = qk / (4 * qr), computed from ggml-common.h defines
    "Q4_0":    QuantTypeInfo("Q4_0",    2,  32,  4, 2,  18, "vec_dot_q4_0_q8_1"),   # qr=2
    "Q4_1":    QuantTypeInfo("Q4_1",    3,  32,  4, 2,  20, "vec_dot_q4_1_q8_1"),   # qr=2
    "Q5_0":    QuantTypeInfo("Q5_0",    6,  32,  4, 2,  22, "vec_dot_q5_0_q8_1"),   # qr=2
    "Q5_1":    QuantTypeInfo("Q5_1",    7,  32,  4, 2,  24, "vec_dot_q5_1_q8_1"),   # qr=2
    "Q8_0":    QuantTypeInfo("Q8_0",    8,  32,  8, 2,  34, "vec_dot_q8_0_q8_1"),   # qr=1
    # K-quants (qk=256)
    "Q2_K":    QuantTypeInfo("Q2_K",   10, 256, 16, 1,  84, "vec_dot_q2_K_q8_1"),   # qr=4
    "Q3_K":    QuantTypeInfo("Q3_K",   11, 256, 16, 1, 110, "vec_dot_q3_K_q8_1"),   # qr=4
    "Q4_K":    QuantTypeInfo("Q4_K",   12, 256, 32, 2, 144, "vec_dot_q4_K_q8_1"),   # qr=2
    "Q5_K":    QuantTypeInfo("Q5_K",   13, 256, 32, 2, 176, "vec_dot_q5_K_q8_1"),   # qr=2
    "Q6_K":    QuantTypeInfo("Q6_K",   14, 256, 32, 1, 210, "vec_dot_q6_K_q8_1"),   # qr=2
    # I-quants (qk=256 except IQ4_NL which is 32)
    "IQ2_XXS": QuantTypeInfo("IQ2_XXS", 16, 256, 16, 2,  66, "vec_dot_iq2_xxs_q8_1"),  # qr=4
    "IQ2_XS":  QuantTypeInfo("IQ2_XS",  17, 256, 16, 2,  74, "vec_dot_iq2_xs_q8_1"),   # qr=4
    "IQ2_S":   QuantTypeInfo("IQ2_S",   22, 256, 16, 2,  82, "vec_dot_iq2_s_q8_1"),    # qr=4
    "IQ3_XXS": QuantTypeInfo("IQ3_XXS", 18, 256, 16, 2,  98, "vec_dot_iq3_xxs_q8_1"),  # qr=4
    "IQ3_S":   QuantTypeInfo("IQ3_S",   21, 256, 16, 2, 110, "vec_dot_iq3_s_q8_1"),    # qr=4
    "IQ1_S":   QuantTypeInfo("IQ1_S",   19, 256,  8, 1,  50, "vec_dot_iq1_s_q8_1"),    # qr=8
    "IQ1_M":   QuantTypeInfo("IQ1_M",   24, 256,  8, 1,  56, "vec_dot_iq1_m_q8_1"),    # qr=8
    "IQ4_NL":  QuantTypeInfo("IQ4_NL",  20,  32,  4, 2,  18, "vec_dot_iq4_nl_q8_1"),   # qr=2
    "IQ4_XS":  QuantTypeInfo("IQ4_XS",  23, 256, 32, 4, 136, "vec_dot_iq4_xs_q8_1"),   # qr=2
}

# sizeof(block_q8_1) = 2*sizeof(half) + 32 = 36 bytes per 32 values
Q8_1_BLOCK_SIZE = 36


def get_quant_info(quant_type: str) -> QuantTypeInfo | None:
    """Look up QuantTypeInfo for a quant type name.

    Accepts both exact names ("Q4_K") and common variants ("Q4_K_M", "Q4_K_S").
    The _M/_S suffix is a GGUF file-level distinction, not a block-level one --
    the underlying block format and vec_dot function are the same.

    Exact match is tried first, so types like "IQ1_S" and "IQ3_S" work correctly
    (those are real type names, not GGUF suffixed variants).
    """
    # Exact match first
    if quant_type in QUANT_TYPES:
        return QUANT_TYPES[quant_type]
    # Then try stripping GGUF-level _M/_S suffixes (e.g., "Q4_K_M" -> "Q4_K")
    if quant_type.endswith(("_M", "_S")):
        base = quant_type.rsplit("_", 1)[0]
        return QUANT_TYPES.get(base)
    return None


# -- llama.cpp path detection --

_LLAMA_CPP_SEARCH_PATHS = [
    Path.home() / "Projects" / "llama-cpp-turboquant",
    Path.home() / "Projects" / "llama-cpp-mainline",
    Path.home() / "Projects" / "llama.cpp",
    Path.home() / "Projects" / "llama-cpp",
    Path("/usr/local/share/llama.cpp"),
]


def find_llama_cpp_path() -> Path | None:
    """Auto-detect a llama.cpp source tree with the required headers.

    Checks a list of common locations. Returns the root of the first tree
    that contains ggml-common.h, common.cuh, and vecdotq.cuh.
    """
    for candidate in _LLAMA_CPP_SEARCH_PATHS:
        if _validate_llama_cpp_path(candidate):
            return candidate
    # Also check LLAMA_CPP_PATH env var
    env_path = os.environ.get("LLAMA_CPP_PATH")
    if env_path and _validate_llama_cpp_path(Path(env_path)):
        return Path(env_path)
    return None


def _validate_llama_cpp_path(root: Path) -> bool:
    """Check that a path contains the required llama.cpp headers."""
    required = [
        root / "ggml" / "src" / "ggml-common.h",
        root / "ggml" / "src" / "ggml-cuda" / "common.cuh",
        root / "ggml" / "src" / "ggml-cuda" / "vecdotq.cuh",
    ]
    return all(f.is_file() for f in required)


def get_llama_cpp_include_dirs(llama_cpp_path: Path) -> list[str]:
    """Return the -I include directories needed to compile against llama.cpp headers.

    The headers have these include relationships:
    - vecdotq.cuh includes common.cuh (same directory)
    - common.cuh includes ggml-common.h (via relative path ../../ggml-common.h
      or via -I pointing to ggml/src/)
    """
    return [
        str(llama_cpp_path / "ggml" / "include"),
        str(llama_cpp_path / "ggml" / "src"),
        str(llama_cpp_path / "ggml" / "src" / "ggml-cuda"),
    ]


# -- Generic kernel generator (uses llama.cpp headers) --

def generate_mmvq_kernel(
    spec: KernelSpec,
    quant_info: QuantTypeInfo,
    llama_cpp_path: Path,
) -> str:
    """Generate a shape-specialized MMVQ kernel for any quantization type.

    This generates a HIP kernel that #includes llama.cpp's existing headers
    to get block struct definitions and vec_dot implementations. The kernel
    structure (grid mapping, inner loop, cross-warp reduction) is the same
    for all types -- only the vec_dot function and type constants differ.

    Args:
        spec: Kernel dimensions and tuning parameters.
        quant_info: Quantization type metadata (qi, vdr, vec_dot function, etc.)
        llama_cpp_path: Path to llama.cpp source root (for header resolution).

    Returns:
        HIP C++ source code as a string.
    """
    N = spec.N
    K = spec.K
    nwarps = spec.nwarps
    rpb = spec.rows_per_block
    warp_size = 32
    threads_per_block = nwarps * warp_size

    qk = quant_info.qk
    qi = quant_info.qi
    vdr = quant_info.vdr

    if K % qk != 0:
        raise ValueError(
            f"K must be divisible by {qk} ({quant_info.name} block size), "
            f"got K={K}. This shape may use padding in llama.cpp."
        )

    blocks_per_row = K // qk
    blocks_per_iter = vdr * nwarps * warp_size // qi
    num_iters = max(1, (blocks_per_row + blocks_per_iter - 1) // blocks_per_iter)
    grid_x = (N + rpb - 1) // rpb

    # The generated kernel calls into llama.cpp's vec_dot functions via headers.
    # All vec_dot_*_q8_1 functions share the same signature:
    #   float vec_dot(const void* vbq, const block_q8_1* bq8_1,
    #                 const int& kbx, const int& iqs)
    # where kbx is the block index into the weight matrix and iqs is the
    # sub-block index within a block.

    kernel = f"""// Generated by kernel-anvil for {quant_info.name} N={N} K={K}
// nwarps={nwarps} rows_per_block={rpb} threads={threads_per_block}
// qk={qk} qi={qi} vdr={vdr}
// blocks_per_row={blocks_per_row} blocks_per_iter={blocks_per_iter} iterations={num_iters}
// Grid: ({grid_x}, 1, 1) Block: ({warp_size}, {nwarps}, 1)
//
// Uses vec_dot from llama.cpp: {quant_info.vec_dot_func}

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Pull in llama.cpp type definitions and vec_dot implementations
#include "ggml-common.h"
#include "common.cuh"
#include "vecdotq.cuh"

extern "C"
__global__ __launch_bounds__({threads_per_block}, 1)
void {spec.name}(
    const void * __restrict__ weights,    // quantized weight matrix [N, K/{qk} blocks]
    const void * __restrict__ input_q8,   // quantized input [K/32 blocks of q8_1]
    float * __restrict__ output,          // output [N]
    const int stride_row                  // blocks per row (= K/{qk})
) {{
    const int tid = threadIdx.y * {warp_size} + threadIdx.x;
    const int row0 = blockIdx.x * {rpb};
"""

    if rpb > 1:
        kernel += f"""
    // Partial sums for {rpb} rows
    float tmp[{rpb}] = {{0.0f}};
"""
    else:
        kernel += """
    float tmp = 0.0f;
"""

    kernel += f"""
    const block_q8_1 * bq8 = (const block_q8_1 *)input_q8;

    // Thread-to-block mapping (same pattern as llama.cpp's mul_mat_vec_q)
    const int iqs = {vdr} * (tid % ({qi}/{vdr}));

    for (int kbx = tid / ({qi}/{vdr}); kbx < {blocks_per_row}; kbx += {blocks_per_iter}) {{
        const int kby = kbx * ({qk}/32);  // QK8_1 = 32
"""

    # Inner loop over rows_per_block
    for i in range(rpb):
        indent = "        "
        t = f"tmp[{i}]" if rpb > 1 else "tmp"
        kernel += f"""
{indent}// Row {i}
{indent}{t} += {quant_info.vec_dot_func}(
{indent}    weights, &bq8[kby], row0 * stride_row + {i} * stride_row + kbx, iqs);
"""

    kernel += "    }\n"

    # Cross-warp reduction
    if nwarps > 1:
        kernel += f"""
    // Cross-warp reduction via shared memory
    __shared__ float smem[{nwarps - 1}][{rpb}][{warp_size}];

    if (threadIdx.y > 0) {{
"""
        for i in range(rpb):
            t = f"tmp[{i}]" if rpb > 1 else "tmp"
            kernel += f"        smem[threadIdx.y - 1][{i}][threadIdx.x] = {t};\n"
        kernel += f"""    }}
    __syncthreads();
    if (threadIdx.y > 0) return;

    // Warp 0 accumulates
    #pragma unroll
    for (int w = 0; w < {nwarps - 1}; ++w) {{
"""
        for i in range(rpb):
            t = f"tmp[{i}]" if rpb > 1 else "tmp"
            kernel += f"        {t} += smem[w][{i}][threadIdx.x];\n"
        kernel += "    }\n"

    # Warp-level reduction
    half_wave = warp_size // 2
    kernel += f"""
    // Warp reduction (wave{warp_size})
    #pragma unroll
    for (int offset = {half_wave}; offset > 0; offset >>= 1) {{
"""
    for i in range(rpb):
        t = f"tmp[{i}]" if rpb > 1 else "tmp"
        kernel += f"        {t} += __shfl_xor({t}, offset, 32);\n"
    kernel += "    }\n"

    # Write output
    kernel += f"""
    // Write output (only lane 0)
    if (threadIdx.x == 0) {{
"""
    for i in range(rpb):
        t = f"tmp[{i}]" if rpb > 1 else "tmp"
        kernel += f"        if (row0 + {i} < {N}) output[row0 + {i}] = {t};\n"
    kernel += "    }\n}\n"

    return kernel


# -- Fallback Q4_K kernel (standalone, no llama.cpp headers needed) --

def generate_q4k_kernel(spec: KernelSpec) -> str:
    """Generate a shape-specialized Q4_K MMVQ HIP kernel.

    This is a standalone GEMV kernel that dequantizes Q4_K weights and
    computes y = W @ x for a specific (N, K) shape.

    The key optimization over the generic kernel: the inner loop count
    is known at compile time, enabling full unrolling and eliminating
    the cross-warp reduction when nwarps=1.

    This is the FALLBACK path for when llama.cpp headers are not available.
    When headers are available, use generate_mmvq_kernel() instead, which
    supports all quantization types.
    """
    N = spec.N
    K = spec.K
    nwarps = spec.nwarps
    rpb = spec.rows_per_block
    warp_size = 32
    threads_per_block = nwarps * warp_size

    if K % 256 != 0:
        raise ValueError(f"K must be divisible by 256 (Q4_K block size), got K={K}. "
                         f"This shape may use padding in llama.cpp.")

    # Q4_K constants
    QK_K = 256      # values per super-block
    QI4_K = 8       # int32s consumed per vec_dot call (qi)
    QR4_K = 2       # vec_dot iterations
    VDR = 2         # vector dot results per call

    blocks_per_row = K // QK_K
    blocks_per_iter = VDR * nwarps * warp_size // QI4_K
    num_iters = max(1, (blocks_per_row + blocks_per_iter - 1) // blocks_per_iter)

    grid_x = (N + rpb - 1) // rpb

    kernel = f"""// Generated by kernel-anvil for {spec.quant_type} N={N} K={K}
// nwarps={nwarps} rows_per_block={rpb} threads={threads_per_block}
// blocks_per_row={blocks_per_row} blocks_per_iter={blocks_per_iter} iterations={num_iters}
// Grid: ({grid_x}, 1, 1) Block: ({warp_size}, {nwarps}, 1)
// FALLBACK: standalone Q4_K kernel (no llama.cpp headers)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Q4_K block structure (144 bytes per 256 values)
struct block_q4_K {{
    __half2 dm;           // super-block scales (d, dmin)
    uint8_t scales[12];   // sub-block scales and mins (6-bit packed)
    uint8_t qs[128];      // 256 x 4-bit quants
}};

// Q8_1 block structure (input, quantized on-the-fly)
struct block_q8_1 {{
    __half2 ds;           // scale and sum (d, sum)
    int8_t qs[32];        // 32 x 8-bit quants
}};

static __device__ __forceinline__ int dp4a(int a, int b, int c) {{
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1200__) || defined(__gfx1201__)
    // RDNA3/4: signed-unsigned dot4 (sudot4)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    // RDNA2/GCN: signed dot4
    return __builtin_amdgcn_sdot4(a, b, c, false);
#endif
}}

extern "C"
__global__ __launch_bounds__({threads_per_block}, 1)
void {spec.name}(
    const void * __restrict__ weights,    // Q4_K weight matrix [N, K/256 blocks]
    const void * __restrict__ input_q8,   // quantized input [K/32 blocks of q8_1]
    float * __restrict__ output,          // output [N]
    const int stride_row                  // blocks per row (= K/256)
) {{
    const int tid = threadIdx.y * {warp_size} + threadIdx.x;
    const int row0 = blockIdx.x * {rpb};
"""

    if rpb > 1:
        kernel += f"""
    // Partial sums for {rpb} rows
    float tmp[{rpb}] = {{0.0f}};
"""
    else:
        kernel += """
    float tmp = 0.0f;
"""

    kernel += f"""
    const block_q4_K * bq4 = (const block_q4_K *)weights;
    const block_q8_1 * bq8 = (const block_q8_1*)input_q8;

    // Each thread processes blocks strided by blocks_per_iter
    const int iqs = {VDR} * (tid % ({QI4_K}/{VDR}));
    const int bq8_offset = {QR4_K} * ((iqs/2) / 4);  // QI8_1/2 = 4

    for (int kbx = tid / ({QI4_K}/{VDR}); kbx < {blocks_per_row}; kbx += {blocks_per_iter}) {{
        const int kby = kbx * ({QK_K}/32);  // QK8_1 = 32
"""

    # Inner loop over rows_per_block
    for i in range(rpb):
        indent = "        "
        kernel += f"""
{indent}// Row {i}
{indent}{{
{indent}    const block_q4_K * bk = bq4 + (row0 + {i}) * stride_row + kbx;

{indent}    // Load 4-bit quants
{indent}    const int * q4 = (const int *)(bk->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
{indent}    int v0 = q4[0];
{indent}    int v1 = q4[4];

{indent}    // Unpack 6-bit scales
{indent}    const uint16_t * scales = (const uint16_t *)bk->scales;
{indent}    uint16_t aux[2];
{indent}    const int j = bq8_offset/2;
{indent}    if (j < 2) {{
{indent}        aux[0] = scales[j+0] & 0x3f3f;
{indent}        aux[1] = scales[j+2] & 0x3f3f;
{indent}    }} else {{
{indent}        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
{indent}        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
{indent}    }}
{indent}    const uint8_t * sc = (const uint8_t *)aux;
{indent}    const uint8_t * m  = sc + 2;

{indent}    // Dot product with input
{indent}    float sumf_d = 0.0f;
{indent}    float sumf_m = 0.0f;
{indent}    #pragma unroll
{indent}    for (int qi = 0; qi < {QR4_K}; ++qi) {{
{indent}        const block_q8_1 * bq8i = bq8 + kby + bq8_offset + qi;
{indent}        float d8 = __low2float(bq8i->ds);
{indent}        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
{indent}        int u0 = q8[0];
{indent}        int u1 = q8[4];

{indent}        int v0i = (v0 >> (4*qi)) & 0x0F0F0F0F;
{indent}        int v1i = (v1 >> (4*qi)) & 0x0F0F0F0F;
{indent}        int dot1 = dp4a(v1i, u1, dp4a(v0i, u0, 0));
{indent}        int dot2 = dp4a(0x01010101, u1, dp4a(0x01010101, u0, 0));

{indent}        sumf_d += d8 * (dot1 * sc[qi]);
{indent}        sumf_m += d8 * (dot2 * m[qi]);
{indent}    }}

{indent}    float2 dm4f = __half22float2(bk->dm);
{indent}    {"tmp[" + str(i) + "]" if rpb > 1 else "tmp"} += dm4f.x * sumf_d - dm4f.y * sumf_m;
{indent}}}
"""

    kernel += "    }\n"

    # Cross-warp reduction
    if nwarps > 1:
        kernel += f"""
    // Cross-warp reduction via shared memory
    __shared__ float smem[{nwarps - 1}][{rpb}][{warp_size}];

    if (threadIdx.y > 0) {{
"""
        for i in range(rpb):
            kernel += f"        smem[threadIdx.y - 1][{i}][threadIdx.x] = {'tmp[' + str(i) + ']' if rpb > 1 else 'tmp'};\n"
        kernel += f"""    }}
    __syncthreads();
    if (threadIdx.y > 0) return;

    // Warp 0 accumulates
    #pragma unroll
    for (int w = 0; w < {nwarps - 1}; ++w) {{
"""
        for i in range(rpb):
            kernel += f"        {'tmp[' + str(i) + ']' if rpb > 1 else 'tmp'} += smem[w][{i}][threadIdx.x];\n"
        kernel += "    }\n"

    # Warp-level reduction (sum across lanes)
    # wave_size/2 = 16 for RDNA (wave32). GCN/CDNA (wave64) would need 32.
    # We only target RDNA2+ (all wave32).
    half_wave = warp_size // 2
    kernel += f"""
    // Warp reduction (wave{warp_size})
    #pragma unroll
    for (int offset = {half_wave}; offset > 0; offset >>= 1) {{
"""
    for i in range(rpb):
        t = f"tmp[{i}]" if rpb > 1 else "tmp"
        kernel += f"        {t} += __shfl_xor({t}, offset, 32);\n"
    kernel += "    }\n"

    # Write output
    kernel += f"""
    // Write output (only lane 0)
    if (threadIdx.x == 0) {{
"""
    for i in range(rpb):
        t = f"tmp[{i}]" if rpb > 1 else "tmp"
        kernel += f"        if (row0 + {i} < {N}) output[row0 + {i}] = {t};\n"
    kernel += "    }\n}\n"

    return kernel


# -- Unified kernel generation entry point --

def generate_kernel(
    spec: KernelSpec,
    llama_cpp_path: Path | None = None,
) -> str:
    """Generate a shape-specialized MMVQ kernel for any supported quant type.

    Uses the generic header-based path when llama.cpp headers are available,
    falling back to the standalone Q4_K generator otherwise.

    Args:
        spec: Kernel dimensions and tuning parameters.
        llama_cpp_path: Path to llama.cpp source. Auto-detected if None.

    Returns:
        HIP C++ source code.

    Raises:
        ValueError: If K is not divisible by the quant block size, or if the
            quant type is unsupported without llama.cpp headers.
    """
    quant_info = get_quant_info(spec.quant_type)

    # Try generic path first
    if llama_cpp_path is None:
        llama_cpp_path = find_llama_cpp_path()

    if llama_cpp_path is not None and quant_info is not None:
        return generate_mmvq_kernel(spec, quant_info, llama_cpp_path)

    # Fallback: standalone Q4_K only
    if spec.quant_type == "Q4_K":
        return generate_q4k_kernel(spec)

    raise ValueError(
        f"No llama.cpp headers found and {spec.quant_type} has no standalone "
        f"fallback. Set LLAMA_CPP_PATH or install llama.cpp source to one of: "
        + ", ".join(str(p) for p in _LLAMA_CPP_SEARCH_PATHS)
    )


# -- Compilation --

def compile_kernel(
    hip_source: str,
    output_path: str,
    arch: str = "gfx1100",
    include_dirs: list[str] | None = None,
) -> str:
    """Compile a HIP kernel to a shared library.

    Args:
        hip_source: HIP C++ source code.
        output_path: Path for the output .so file.
        arch: GPU architecture target (default gfx1100).
        include_dirs: Additional -I include directories (for llama.cpp headers).

    Returns:
        Path to the compiled .so file.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".hip", delete=False) as f:
        f.write(hip_source)
        src_path = f.name

    try:
        cmd = [
            "hipcc",
            "--offload-arch=" + arch,
            "-shared", "-fPIC",
            "-O3",
        ]
        # Add include directories for llama.cpp headers
        if include_dirs:
            for d in include_dirs:
                cmd.extend(["-I", d])
        # Architecture macro definitions for RDNA detection in the headers
        if "gfx1100" in arch or "gfx1101" in arch or "gfx1102" in arch:
            cmd.append("-DRDNA3_0")
            cmd.append("-DRDNA3")
        elif "gfx1150" in arch or "gfx1151" in arch:
            cmd.append("-DRDNA3_5")
            cmd.append("-DRDNA3")
        elif "gfx1200" in arch or "gfx1201" in arch:
            cmd.append("-DRDNA4")
        elif "gfx1030" in arch or "gfx1031" in arch or "gfx1032" in arch:
            cmd.append("-DRDNA2")
        elif "gfx1010" in arch or "gfx1012" in arch:
            cmd.append("-DRDNA1")
        # Required defines for ggml headers to use HIP path
        cmd.append("-DGGML_COMMON_DECL_HIP")
        cmd.append("-DGGML_COMMON_IMPL_HIP")
        cmd.append("-DGGML_USE_HIP")
        cmd.append("-DMMVQ_MAX_BATCH_SIZE=8")
        cmd.extend(["-o", output_path, src_path])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"hipcc failed:\n{result.stderr}")
        return output_path
    finally:
        os.unlink(src_path)


def _heuristic_config(quant_info: QuantTypeInfo, n: int, k: int) -> tuple[int, int]:
    """Pick nwarps and rows_per_block heuristically for a given shape.

    Returns:
        (nwarps, rows_per_block) tuple.
    """
    blocks_per_row = k // quant_info.qk

    # K-quants and large blocks: more parallelism needed
    if quant_info.qk == 256:
        if blocks_per_row < 64:
            return (2, 2)
        else:
            return (4, 1)
    else:
        # Basic quants (qk=32): many more blocks per row
        if blocks_per_row < 512:
            return (2, 2)
        else:
            return (4, 1)


def generate_model_kernels(
    shapes: dict[tuple[str, int, int], int],
    output_dir: str,
    arch: str = "gfx1100",
    optimal_configs: dict | None = None,
    llama_cpp_path: Path | str | None = None,
) -> list[tuple[KernelSpec, str]]:
    """Generate and compile optimized kernels for all shapes in a model.

    Args:
        shapes: Dict of (quant_type, N, K) -> count from GGUF parser.
        output_dir: Directory for compiled .so files.
        arch: GPU architecture.
        optimal_configs: Optional dict of (quant_type, N, K) -> {"nwarps": int, "rows_per_block": int}.
            If not provided, uses heuristic defaults.
        llama_cpp_path: Path to llama.cpp source. Auto-detected if None.

    Returns:
        List of (KernelSpec, so_path) tuples.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    # Resolve llama.cpp path once
    if llama_cpp_path is not None:
        llama_cpp_path = Path(llama_cpp_path)
    else:
        llama_cpp_path = find_llama_cpp_path()

    # Build include dirs if we have llama.cpp headers
    include_dirs = None
    if llama_cpp_path is not None:
        include_dirs = get_llama_cpp_include_dirs(llama_cpp_path)

    for (qt, n, k), count in shapes.items():
        quant_info = get_quant_info(qt)

        # Skip unsupported types
        if quant_info is None:
            # F16/F32/BF16 don't use the MMVQ path
            continue

        # Without llama.cpp headers, only Q4_K works (fallback path)
        if llama_cpp_path is None and qt != "Q4_K":
            continue

        # Get optimal config or use heuristic
        if optimal_configs and (qt, n, k) in optimal_configs:
            cfg = optimal_configs[(qt, n, k)]
            nwarps = cfg["nwarps"]
            rpb = cfg.get("rows_per_block", 1)
        else:
            nwarps, rpb = _heuristic_config(quant_info, n, k)

        spec = KernelSpec(quant_type=qt, N=n, K=k, nwarps=nwarps, rows_per_block=rpb)

        try:
            hip_src = generate_kernel(spec, llama_cpp_path)
        except ValueError as e:
            print(f"  Skipping {spec.name}: {e}")
            continue

        so_path = os.path.join(output_dir, f"{spec.name}.so")
        try:
            compile_kernel(hip_src, so_path, arch, include_dirs)
            results.append((spec, so_path))
        except RuntimeError as e:
            print(f"  Failed to compile {spec.name}: {e}")

    return results
