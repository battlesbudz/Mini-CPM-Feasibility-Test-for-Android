#!/usr/bin/env python3
"""Fetch the pinned llama.cpp-omni fork for the MiniCPM-V 4.7 trial.

The pin is the exact submodule commit OpenBMB's MiniCPM-V-Apps demos build
against (tc-mb/llama.cpp-omni, branch ios-omni): it carries the LES / MoE-SSD
expert-streaming core (src/llama-moe-*) and the MiniCPM-V 4.7 mtmd support.
Never modify the upstream checkout; refuse to run over local changes.
"""
import pathlib, subprocess, shutil
ROOT = pathlib.Path(__file__).resolve().parents[1]
PIN = "96bbf61710d44421ea54e63f5194f9b17c20d877"
dest = ROOT / "vendor/llama.cpp-omni-v47"
def git(*args): return subprocess.check_output(["git", *args], text=True).strip()
if not dest.exists():
    dest.parent.mkdir(exist_ok=True)
    subprocess.run(["git", "clone", "--filter=blob:none", "https://github.com/tc-mb/llama.cpp-omni.git", str(dest)], check=True)
if git("-C", str(dest), "status", "--porcelain"):
    raise SystemExit("Dependency has local changes; refusing to overwrite them")
subprocess.run(["git", "-C", str(dest), "checkout", "--detach", PIN], check=True)
assert git("-C", str(dest), "rev-parse", "HEAD") == PIN
assets = ROOT / "v47/src/main/assets"
assets.mkdir(parents=True, exist_ok=True)
shutil.copyfile(dest / "LICENSE", assets / "NATIVE_LICENSE.txt")
print("Prepared MiniCPM-V 4.7 native runtime", PIN)
