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
"""
import pathlib, shutil, subprocess, tempfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
PIN = "96bbf61710d44421ea54e63f5194f9b17c20d877"
dest = ROOT / "vendor/llama.cpp-omni-v47"
PATCH = ROOT / "v47/patches/ggml-cpu-alloc-only-buft.patch"
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
    # True iff the working tree differs from HEAD by exactly our patch.
    # Byte-exact: rebuild HEAD + patch in a scratch dir and compare, so an
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
        if subprocess.run(["git", "apply", str(PATCH)], cwd=td).returncode != 0:
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
run("-C", str(dest), "apply", str(PATCH))
assets = ROOT / "v47/src/main/assets"
assets.mkdir(parents=True, exist_ok=True)
shutil.copyfile(dest / "LICENSE", assets / "NATIVE_LICENSE.txt")
print("Prepared MiniCPM-V 4.7 native runtime", PIN, "+ patch", PATCH.name)
