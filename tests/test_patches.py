"""Guards on the llama.cpp integration patch (patches/).

Regression tests for GitHub issue #13: the mmvq-smithy.patch hunk that edits
calc_rows_per_block carried only 3 context lines, and that context (the
``default: return 1; } } return 1; }`` tail) is byte-identical at the end of
calc_nwarps, the function directly above. When upstream line numbers drift,
git apply places the insertion inside calc_nwarps, where small_k is not a
parameter, and the build fails with "use of undeclared identifier 'small_k'".

The fix pins the hunk with enough leading context to include the
calc_rows_per_block signature, so the patch either lands in the right
function or fails loudly. These tests keep that property, and keep the
README manual instructions and apply.sh safety check in sync with it.
"""

import re
from pathlib import Path

PATCHES = Path(__file__).resolve().parent.parent / "patches"

CALC_RPB_SIGNATURE = (
    "int calc_rows_per_block(int ncols_dst, int table_id, bool small_k"
)
INSERT_MARKER = "if (small_k && ncols_dst == 1) {"


def _hunks(patch_text):
    """Split a unified diff into hunks: list of (header, [lines])."""
    hunks = []
    current = None
    for line in patch_text.splitlines():
        if line.startswith("@@"):
            current = (line, [])
            hunks.append(current)
        elif current is not None and line[:1] in (" ", "+", "-"):
            current[1].append(line)
    return hunks


def _load_patch():
    return (PATCHES / "mmvq-smithy.patch").read_text()


def test_small_k_hunk_context_includes_signature():
    """The hunk inserting into calc_rows_per_block must carry the function
    signature as *context* (a line starting with ' '), so it cannot be
    placed into calc_nwarps, whose tail is otherwise identical."""
    for header, lines in _hunks(_load_patch()):
        added = [l for l in lines if l.startswith("+")]
        if any(INSERT_MARKER in l for l in added):
            context = [l for l in lines if l.startswith(" ")]
            sig_in_context = any(CALC_RPB_SIGNATURE in l for l in context)
            assert sig_in_context, (
                "the small_k hunk does not include the calc_rows_per_block "
                "signature as context; git apply can misplace it into "
                "calc_nwarps (issue #13)"
            )
            insert_idx = next(
                i for i, l in enumerate(lines) if l.startswith("+") and INSERT_MARKER in l
            )
            sig_idx = next(
                i for i, l in enumerate(lines)
                if l.startswith(" ") and CALC_RPB_SIGNATURE in l
            )
            assert sig_idx < insert_idx, (
                "signature context must precede the insertion"
            )
            break
    else:
        raise AssertionError("no hunk inserts the small_k block")


def test_readme_manual_steps_match_patch():
    """README's hand-apply snippet must stay in sync with the real patch:
    same return expression, and it must warn about the identical
    calc_nwarps tail (issue #13 came from following stale instructions)."""
    readme = (PATCHES / "README.md").read_text()
    patch = _load_patch()

    added = "\n".join(
        l[1:] for _, lines in _hunks(patch) for l in lines if l.startswith("+")
    )
    rpb_return = re.search(r"if \(small_k && ncols_dst == 1\) \{\s*\n\s*return (\w+);", added)
    assert rpb_return, "patch no longer contains the small_k insertion"
    assert f"return {rpb_return.group(1)};" in readme, (
        "README manual snippet returns a different value than the patch"
    )
    assert "return 2;" not in readme or rpb_return.group(1) == "2", (
        "README still shows the stale `return 2;` snippet"
    )
    assert "calc_nwarps" in readme, (
        "README must warn that calc_nwarps ends with an identical tail"
    )
    assert CALC_RPB_SIGNATURE in readme, (
        "README must show the full signature so users edit the right function"
    )


def test_apply_sh_verifies_placement():
    """apply.sh must check, after patching, that the insertion sits inside
    calc_rows_per_block, and restore the original file if it does not."""
    script = (PATCHES / "apply.sh").read_text()
    assert "calc_rows_per_block" in script, (
        "apply.sh has no post-apply placement verification"
    )
    assert "small_k && ncols_dst" in script, (
        "apply.sh does not look for the inserted line when verifying"
    )
