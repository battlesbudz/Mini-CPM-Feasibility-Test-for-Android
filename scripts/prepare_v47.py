#!/usr/bin/env python3
"""Fetch the pinned llama.cpp-omni fork for the MiniCPM-V 4.7 trial.

The pin is the exact submodule commit OpenBMB's MiniCPM-V-Apps demos build
against (tc-mb/llama.cpp-omni, branch ios-omni): it carries the LES / MoE-SSD
expert-streaming core (src/llama-moe-*) and the MiniCPM-V 4.7 mtmd support.
The upstream checkout is never edited by hand. The changes the trial needs
on top of the pin (a CPU alloc-only buffer type for the LES load path,
plus the loader changes that make the loader use it for CPU layers, plus
a telemetry-only lookahead-accuracy counter in the slot runtime; see
the patch header for why) live in v47/patches/ and are applied mechanically
below, after the pin is checked out. Any other local changes to the
checkout are refused, exactly as before.

Build 24 adds a second patch to the list: the item-2 frequency-aware
admission policies in the slot runtime (pin the prewarm set; LFU
admission with a probationary segment), env-gated and default off; see
moe-slot-admission.patch's own header. It touches only
src/llama-moe-slot.cpp, in regions disjoint from the first patch's
hunks there, and the integrity check below covers the union of both
patches' files byte-exactly, exactly as it covered the one patch.

Build 27 adds a third patch: the item-5 tail-skip in the slot runtime
(a DSpark verify-batch expert routed only at the last rank is not
installed; its contribution is zeroed for that step through a reserved
zero slot), env-gated (LLAMA_MOE_TAIL_SKIP) and default off; see
moe-slot-tail-skip.patch's own header. It also touches only
src/llama-moe-slot.cpp, in regions disjoint from both earlier patches'
hunks there (it applies onto the pristine pin standalone, and in
either order relative to the admission patch), so PATCHED_FILES is
unchanged and the same byte-exact check covers all three.
"""
import pathlib, shutil, subprocess, tempfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
PIN = "96bbf61710d44421ea54e63f5194f9b17c20d877"
dest = ROOT / "vendor/llama.cpp-omni-v47"
PATCHES = [
    ROOT / "v47/patches/ggml-cpu-alloc-only-buft.patch",
    ROOT / "v47/patches/moe-slot-admission.patch",
    ROOT / "v47/patches/moe-slot-tail-skip.patch",
]
PATCHED_FILES = [
    "ggml/include/ggml-backend.h",
    "ggml/src/ggml-backend.cpp",
    "ggml/src/ggml-cpu/ggml-cpu.cpp",
    "src/llama-model.cpp",
    "src/llama-moe-slot.cpp",
]
def git(*args): return subprocess.check_output(["git", *args], text=True).strip()
def run(*args): subprocess.run(["git", *args], check=True)
def dirty_paths():
    # No strip(): porcelain lines start with two status columns and a space,
    # and stripping the output would eat the first line's leading column.
    out = subprocess.check_output(["git", "-C", str(dest), "status", "--porcelain"], text=True)
    return {line[3:] for line in out.splitlines() if line}
def patch_is_applied():
    # True iff the working tree differs from HEAD by exactly our patches.
    # Byte-exact: rebuild HEAD + patches in a scratch dir and compare, so an
    # unrelated edit anywhere in the touched files (even outside the patch
    # hunks) is still treated as foreign and refused, never silently reset.
    if dirty_paths() != set(PATCHED_FILES):
        return False
    with tempfile.TemporaryDirectory() as td:
        for rel in PATCHED_FILES:
            blob = subprocess.check_output(["git", "-C", str(dest), "show", f"HEAD:{rel}"])
            scratch = pathlib.Path(td, rel)
            scratch.parent.mkdir(parents=True, exist_ok=True)
            scratch.write_bytes(blob)
        for patch in PATCHES:
            if subprocess.run(["git", "apply", str(patch)], cwd=td).returncode != 0:
                return False
        return all(pathlib.Path(td, rel).read_bytes() == (dest / rel).read_bytes()
                   for rel in PATCHED_FILES)
if not dest.exists():
    dest.parent.mkdir(exist_ok=True)
    subprocess.run(["git", "clone", "--filter=blob:none", "https://github.com/tc-mb/llama.cpp-omni.git", str(dest)], check=True)
if dirty_paths():
    if not patch_is_applied():
        raise SystemExit("Dependency has local changes; refusing to overwrite them")
    # A previous run left exactly our patch applied. Reset those files so
    # the checkout + apply below reproduces the same state deterministically.
    run("-C", str(dest), "checkout", "--", *PATCHED_FILES)
subprocess.run(["git", "-C", str(dest), "checkout", "--detach", PIN], check=True)
assert git("-C", str(dest), "rev-parse", "HEAD") == PIN
for patch in PATCHES:
    run("-C", str(dest), "apply", str(patch))
assets = ROOT / "v47/src/main/assets"
assets.mkdir(parents=True, exist_ok=True)
shutil.copyfile(dest / "LICENSE", assets / "NATIVE_LICENSE.txt")
print("Prepared MiniCPM-V 4.7 native runtime", PIN, "+ patches",
      ", ".join(p.name for p in PATCHES))
