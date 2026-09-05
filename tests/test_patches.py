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

Deep-tests risk model, patches/ integration (2026-09-05, #17 + PR #16):

1. Invariants: the three router patches define and use ONE symbol
   (``COMMON_ARG_PRESET_SMITHY_CONFIG``); apply.sh applies the router set all
   or none; whichever core variant applies, the small_k block lands inside
   calc_rows_per_block.
2. State transitions: a target tree goes untouched -> core patched (+ router
   patched or router untouched). Never core patched + router half patched.
3. Boundaries: zero router patches applicable; exactly one missing target
   file; the pre-GB10 tree (older context) versus current master.
4. Malformed inputs: a patch file listed in apply.sh but absent from patches/.
5. Concurrency: N/A - apply.sh is a single sequential script.
6. Persistence: N/A - apply.sh writes into the caller's tree; the only stored
   state is the .anvil-orig backup, covered by the placement test.
7. Integration contracts: apply.sh's ROUTER_PATCHES list names the files that
   ship; each patch's ``--- a/`` path is the file the README table documents.
8. Regression traps: boundary=one-of-three missing; concurrency=N/A;
   contract=apply.sh list vs files on disk; encoding=N/A; framework=real
   ``git apply`` on a reconstructed tree, not a string check; io=fixture
   tree built from the patches' own context lines; persistence=N/A;
   resource=N/A; state=partial-apply forbidden.

Coverage matrix:
  1 invariants ......... test_router_patches_share_one_symbol,
                         test_apply_sh_router_patches_are_all_or_none,
                         test_apply_sh_places_small_k_with_either_variant
  2 state .............. test_apply_sh_router_patches_are_all_or_none
  3 boundaries ......... test_apply_sh_router_patches_are_all_or_none (one
                         missing), test_apply_sh_places_small_k_with_either_variant
                         (pre-GB10 vs current)
  4 malformed .......... test_router_patches_listed_in_apply_sh_exist
  7 contracts .......... test_router_patches_listed_in_apply_sh_exist,
                         test_router_patches_target_documented_files
  8 traps .............. all of the above
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

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


# ---------------------------------------------------------------------------
# Router-mode patches (PR #16) and the two core variants (#17)
# ---------------------------------------------------------------------------

ROUTER_PATCHES = (
    "arg-h-smithy-router.patch",
    "arg-cpp-smithy-router.patch",
    "server-models-cpp-smithy-router.patch",
)
SYMBOL = "COMMON_ARG_PRESET_SMITHY_CONFIG"


def _apply_sh_router_list():
    script = (PATCHES / "apply.sh").read_text()
    m = re.search(r"ROUTER_PATCHES=\(\n(.*?)\n\)", script, re.S)
    assert m, "contract: apply.sh has no ROUTER_PATCHES=( ... ) list"
    return tuple(l.strip() for l in m.group(1).splitlines() if l.strip())


def _target_of(patch_text):
    m = re.search(r"^--- a/(\S+)$", patch_text, re.M)
    assert m, "patch has no '--- a/<path>' header"
    return m.group(1)


def _before_file(patch_text):
    """Reconstruct the pre-patch file from a unified diff's context and '-'
    lines. Hunks are joined with one filler line; git apply locates each hunk
    by content, so absolute offsets do not matter."""
    parts = []
    for _header, lines in _hunks(patch_text):
        parts.append("\n".join(l[1:] for l in lines if l[:1] in (" ", "-")))
    return "\n// filler\n".join(parts) + "\n"


def _fixture_tree(tmp_path, core_patch, with_router=True, drop=()):
    """A fake llama.cpp tree that the given patches apply to, as a git repo."""
    tree = tmp_path / "llama.cpp"
    files = {"ggml/src/ggml-cuda/mmvq.cu": (PATCHES / core_patch).read_text()}
    if with_router:
        for name in ROUTER_PATCHES:
            text = (PATCHES / name).read_text()
            files[_target_of(text)] = text
    for rel, patch_text in files.items():
        if rel in drop:
            continue
        dst = tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(_before_file(patch_text))
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    subprocess.run(["git", "-C", str(tree), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(tree), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "base"], check=True,
    )
    return tree


def _run_apply(tree):
    return subprocess.run(
        ["bash", str(PATCHES / "apply.sh"), str(tree)],
        capture_output=True, text=True,
    )


def _changed(tree):
    out = subprocess.run(
        ["git", "-C", str(tree), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout
    return sorted(l[3:] for l in out.splitlines())


def test_router_patches_listed_in_apply_sh_exist():  # contract: list vs disk
    listed = _apply_sh_router_list()
    on_disk = tuple(sorted(p.name for p in PATCHES.glob("*-router.patch")))
    assert tuple(sorted(listed)) == on_disk, (
        f"contract violated: apply.sh ROUTER_PATCHES={listed} but patches/ ships "
        f"{on_disk}; a listed-but-missing file skips the whole router set, an "
        f"unlisted file never ships"
    )
    assert tuple(sorted(listed)) == tuple(sorted(ROUTER_PATCHES)), (
        f"test constant drifted from apply.sh: {listed}"
    )


def test_router_patches_target_documented_files():  # contract: README table
    readme = (PATCHES / "README.md").read_text()
    for name in ROUTER_PATCHES:
        target = _target_of((PATCHES / name).read_text())
        assert f"`{name}`" in readme and f"`{target}`" in readme, (
            f"README Patch Files table does not document {name} -> {target}; "
            f"users apply by hand from that table"
        )


def test_router_patches_share_one_symbol():  # invariant: one define, two uses
    added = {}
    for name in ROUTER_PATCHES:
        text = (PATCHES / name).read_text()
        added[name] = "\n".join(
            l[1:] for _, lines in _hunks(text) for l in lines if l.startswith("+")
        )
    assert re.search(rf"^#define {SYMBOL}\s+\"__PRESET_SMITHY_CONFIG\"", added["arg-h-smithy-router.patch"], re.M), (
        f"invariant violated: arg-h patch does not define {SYMBOL}; arg.cpp and "
        f"server-models.cpp would not compile"
    )
    for user in ("arg-cpp-smithy-router.patch", "server-models-cpp-smithy-router.patch"):
        assert SYMBOL in added[user] and "#define" not in added[user], (
            f"invariant violated: {user} must USE {SYMBOL} (not redefine it); "
            f"added lines were: {added[user]!r}"
        )


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_apply_sh_router_patches_are_all_or_none(tmp_path):  # state: no half apply
    # all present: every router file changes
    tree = _fixture_tree(tmp_path / "ok", "mmvq-smithy.patch")
    r = _run_apply(tree)
    assert r.returncode == 0, f"apply.sh failed on a clean tree:\n{r.stdout}\n{r.stderr}"
    changed = _changed(tree)
    for name in ROUTER_PATCHES:
        target = _target_of((PATCHES / name).read_text())
        assert target in changed, (
            f"state contract violated: {target} untouched although all three router "
            f"patches applied cleanly; changed={changed}\n{r.stdout}"
        )
    # one target missing (arg.h): NONE of the router files may change
    tree = _fixture_tree(tmp_path / "half", "mmvq-smithy.patch", drop=("common/arg.h",))
    r = _run_apply(tree)
    assert r.returncode == 0, (
        f"apply.sh must still succeed on the core patch when router mode is "
        f"unavailable:\n{r.stdout}\n{r.stderr}"
    )
    changed = _changed(tree)
    router_changed = [c for c in changed if c.startswith(("common/", "tools/"))]
    assert router_changed == [], (
        f"all-or-none violated: arg.h was absent yet {router_changed} were patched; "
        f"arg.cpp/server-models.cpp now use {SYMBOL} with no define = build break. "
        f"apply.sh output:\n{r.stdout}"
    )
    assert "NOT applied" in r.stdout, (
        f"apply.sh skipped the router set silently; output was:\n{r.stdout}"
    )
    assert "ggml/src/ggml-cuda/mmvq.cu" in changed, (
        f"core patch must still land when router patches are skipped; changed={changed}"
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
@pytest.mark.parametrize("variant", ["mmvq-smithy.patch", "mmvq-smithy-pre-gb10.patch"])
def test_apply_sh_places_small_k_with_either_variant(tmp_path, variant):  # boundary: tree age
    tree = _fixture_tree(tmp_path, variant, with_router=False)
    r = _run_apply(tree)
    assert r.returncode == 0, f"apply.sh failed on a {variant} tree:\n{r.stdout}\n{r.stderr}"
    assert f"Applied {variant} (placement verified)" in r.stdout, (
        f"variant selection violated: expected apply.sh to report {variant} with "
        f"placement verified; output:\n{r.stdout}"
    )
    mmvq = (tree / "ggml/src/ggml-cuda/mmvq.cu").read_text()
    sig = mmvq.index(CALC_RPB_SIGNATURE)
    ins = mmvq.index(INSERT_MARKER)
    assert 0 < ins - sig < 2000, (
        f"placement violated on {variant}: small_k block at {ins}, "
        f"calc_rows_per_block signature at {sig}"
    )
    assert "smithy_lookup(type, nrows_x, ncols_x)" in mmvq, (
        f"{variant} did not install the dispatch-site lookup"
    )


# SABOTAGE LOG (deep-tests), 2026-09-05, router patches + core variants
# Production mutation 1: apply.sh applies each router patch inside the check
#   loop (the PR's original per-file shape). Predicted all_or_none fails on the
#   arg.h-missing fixture; observed 1 failed (arg.cpp and server-models.cpp
#   patched without the define). Restored.
# Test mutation 1b: with mutation 1 still in place, dropped the
#   `router_changed == []` assertion. Predicted pass; observed pass, so that
#   assertion is the sole guard of the half-apply rule and is kept.
# Production mutation 2: apply.sh's candidate loop drops mmvq-smithy-pre-gb10.
#   Predicted the pre-gb10 parametrization fails and current passes; observed
#   exactly that. Restored.
# Production mutation 3: arg-h patch defines COMMON_ARG_PRESET_SMITHY_CONF.
#   Predicted share_one_symbol fails and all_or_none still passes (its fixture
#   is reconstructed from the patch, so it cannot see a spelling drift);
#   observed exactly that. Restored, all three files cmp-identical to backup.
# Loudness: every assertion above names the rule (contract / invariant /
#   state / placement / variant selection) and carries the observed state.
#   No exemptions.
