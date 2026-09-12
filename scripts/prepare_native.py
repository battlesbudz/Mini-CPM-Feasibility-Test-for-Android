#!/usr/bin/env python3
"""Fetch the pinned dependency without modifying Jarvis V2."""
import pathlib, subprocess, shutil
ROOT = pathlib.Path(__file__).resolve().parents[1]
PIN = "64d092c60db4b4ee45768476bd752f03fdcc98ea"
dest = ROOT / "vendor/llama.cpp-omni"
def git(*args): return subprocess.check_output(["git", *args], text=True).strip()
if not dest.exists():
    dest.parent.mkdir(exist_ok=True)
    subprocess.run(["git", "clone", "--filter=blob:none", "https://github.com/tc-mb/llama.cpp-omni.git", str(dest)], check=True)
if git("-C", str(dest), "status", "--porcelain"):
    raise SystemExit("Dependency has local changes; refusing to overwrite them")
subprocess.run(["git", "-C", str(dest), "checkout", "--detach", PIN], check=True)
assert git("-C", str(dest), "rev-parse", "HEAD") == PIN
assets = ROOT / "app/src/main/assets"
assets.mkdir(parents=True, exist_ok=True)
shutil.copyfile(dest / "tools/omni/assets/default_ref_audio/default_ref_audio.wav", assets / "reference.wav")
shutil.copyfile(dest / "LICENSE", assets / "NATIVE_LICENSE.txt")
print("Prepared native runtime", PIN)
