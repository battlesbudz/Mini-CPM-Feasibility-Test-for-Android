#!/usr/bin/env python3
"""Fetch the exact official runner revision; never modify the upstream checkout."""
import pathlib,subprocess,shutil
ROOT=pathlib.Path(__file__).resolve().parents[1]
PIN='ec9d1fdd9cc18643c5e161b65b8053f6f6f34a4b'
dest=ROOT/'vendor/llama.cpp-liquid'
def git(*args):return subprocess.check_output(['git',*args],text=True).strip()
if not dest.exists():
 dest.parent.mkdir(exist_ok=True)
 subprocess.run(['git','clone','--filter=blob:none','https://github.com/tdakhran/llama.cpp.git',str(dest)],check=True)
if git('-C',str(dest),'status','--porcelain'):raise SystemExit('Dependency has local changes; refusing to overwrite')
subprocess.run(['git','-C',str(dest),'checkout','--detach',PIN],check=True)
assert git('-C',str(dest),'rev-parse','HEAD')==PIN
shutil.copyfile(dest/'LICENSE',ROOT/'liquid/src/main/assets/NATIVE_LICENSE.txt')
print('Prepared Liquid runner',PIN)
