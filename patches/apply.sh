#!/bin/bash
# Apply kernel-anvil runtime config patch to llama.cpp
# Usage: ./apply.sh /path/to/llama.cpp

set -e

LLAMA_CPP="${1:?Usage: $0 /path/to/llama.cpp}"
LLAMA_CPP="$(cd "$LLAMA_CPP" && pwd)" || { echo "Error: $1 is not a directory"; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMVQ="$LLAMA_CPP/ggml/src/ggml-cuda/mmvq.cu"

if [ ! -f "$MMVQ" ]; then
    echo "Error: $LLAMA_CPP doesn't look like a llama.cpp source tree"
    echo "Expected: ggml/src/ggml-cuda/mmvq.cu"
    exit 1
fi

manual_steps() {
    echo "Manual steps:"
    echo "  1. smithy-config.h is already copied"
    echo "  2. Add '#include \"smithy-config.h\"' to the top of ggml/src/ggml-cuda/mmvq.cu"
    echo "  3. See patches/README.md for the two small code changes needed"
}

# Copy smithy-config.h
cp "$SCRIPT_DIR/smithy-config.h" "$LLAMA_CPP/ggml/src/ggml-cuda/smithy-config.h"
echo "Copied smithy-config.h"

# Apply mmvq.cu patch (keep a backup so a bad apply never leaves a broken tree)
cp "$MMVQ" "$MMVQ.anvil-orig"
cd "$LLAMA_CPP"
# Two variants of the same hook. Upstream commit 25ae3a9b3 (2026-08-18, GB10
# MMVQ table) changed the calc_rows_per_block context our hunk anchors on, so
# trees from before it (including the ROCm/TheRock fork as of 2026-09) need the
# pre-gb10 file. Try current first; whichever applies cleanly is used.
PATCH=""
for candidate in mmvq-smithy.patch mmvq-smithy-pre-gb10.patch; do
    if git apply --check "$SCRIPT_DIR/$candidate" 2>/dev/null; then
        PATCH="$candidate"
        break
    fi
done
if [ -z "$PATCH" ]; then
    rm -f "$MMVQ.anvil-orig"
    echo "Neither patch variant applies cleanly (llama.cpp version mismatch)."
    manual_steps
    exit 1
fi
git apply "$SCRIPT_DIR/$PATCH"
echo "Applied $PATCH"

# Verify placement (issue #13): calc_nwarps, directly above calc_rows_per_block,
# ends with an identical 'return 1; }' tail. If context drift ever lands the
# small_k block outside calc_rows_per_block, small_k/nwarps are undeclared and
# the build breaks. Check the insertion sits below the right signature.
if awk '
    /int calc_rows_per_block\(int ncols_dst, int table_id, bool small_k/ { sig = NR }
    /if \(small_k && ncols_dst == 1\) \{/                                { ins = NR }
    /smithy_lookup\(type, nrows_x, ncols_x\)/                            { lut = NR }
    END { exit !(sig && ins && lut && ins > sig && ins < sig + 40) }
' "$MMVQ"; then
    rm -f "$MMVQ.anvil-orig"
    echo "Applied $PATCH (placement verified)"
else
    mv "$MMVQ.anvil-orig" "$MMVQ"
    echo "Error: patch applied but the small_k block did not land inside"
    echo "calc_rows_per_block (your llama.cpp revision has drifted)."
    echo "mmvq.cu has been restored to its original state."
    manual_steps
    exit 1
fi

echo ""
echo "Done! Now rebuild llama.cpp with HIP:"
echo "  cd $LLAMA_CPP"
echo "  cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build build --config Release -j\$(nproc)"
echo ""
echo "Then run with kernel-anvil configs:"
echo "  kernel-anvil gguf-optimize model.gguf"
echo "  SMITHY_CONFIG=~/.cache/smithy/model.json ./build/bin/llama-server -m model.gguf -ngl 999"
