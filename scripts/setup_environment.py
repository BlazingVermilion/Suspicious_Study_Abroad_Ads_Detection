#!/usr/bin/env python3
"""
Set up the Python environment for this project.

This script must be run from inside the target virtual environment, for example:

  .\.venv\Scripts\python.exe scripts\setup_environment.py --device gpu --cuda cu128

Why this exists:
  requirements.txt intentionally does not install torch. PyTorch must be installed
  separately because CPU and CUDA builds come from different package indexes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


CUDA_INDEXES = {
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "nightly-cu128": "https://download.pytorch.org/whl/nightly/cu128",
    "nightly-cu129": "https://download.pytorch.org/whl/nightly/cu129",
}


def root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def run(cmd: List[str], dry_run: bool = False, allow_failure: bool = False) -> bool:
    printable = " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd)
    print("\n" + "=" * 100)
    print(printable)
    print("=" * 100)
    if dry_run:
        return True
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        if allow_failure:
            print(f"[warn] command failed with exit code {proc.returncode}; continuing.")
            return False
        raise SystemExit(proc.returncode)
    return True


def pip_cmd(*args: str) -> List[str]:
    return [sys.executable, "-m", "pip", *args]


def verify_torch(dry_run: bool = False) -> None:
    code = (
        "import torch; "
        "print('torch:', torch.__version__); "
        "print('cuda build:', torch.version.cuda); "
        "print('cuda available:', torch.cuda.is_available()); "
        "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
    )
    run([sys.executable, "-c", code], dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install base requirements and a CPU/GPU PyTorch build.")
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True, help="Install CPU or CUDA/GPU PyTorch.")
    parser.add_argument("--cuda", choices=sorted(CUDA_INDEXES), default="cu128", help="CUDA wheel index for --device gpu.")
    parser.add_argument("--skip-base", action="store_true", help="Do not install requirements.txt.")
    parser.add_argument("--reset-torch", action="store_true", default=True, help="Uninstall torch/torchvision/torchaudio before installing.")
    parser.add_argument("--no-reset-torch", dest="reset_torch", action="store_false", help="Do not uninstall existing torch packages first.")
    parser.add_argument("--allow-nightly-fallback", action="store_true", help="If stable CUDA install fails, try nightly cu128 then nightly cu129.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    root = root_dir()
    req = root / "requirements.txt"
    if not req.exists():
        raise FileNotFoundError(f"Missing requirements.txt: {req}")

    print(f"[python] {sys.executable}")
    print(f"[root]   {root}")
    print(f"[device] {args.device}")

    run(pip_cmd("install", "--upgrade", "pip", "setuptools", "wheel"), dry_run=args.dry_run)

    if not args.skip_base:
        run(pip_cmd("install", "-r", str(req)), dry_run=args.dry_run)

    if args.reset_torch:
        run(pip_cmd("uninstall", "-y", "torch", "torchvision", "torchaudio"), dry_run=args.dry_run, allow_failure=True)

    if args.device == "cpu":
        run(pip_cmd("install", "torch", "torchvision", "torchaudio"), dry_run=args.dry_run)
    else:
        primary_index = CUDA_INDEXES[args.cuda]
        ok = run(
            pip_cmd("install", "torch", "torchvision", "torchaudio", "--index-url", primary_index),
            dry_run=args.dry_run,
            allow_failure=args.allow_nightly_fallback,
        )
        if not ok and args.allow_nightly_fallback:
            for fallback in ["nightly-cu128", "nightly-cu129"]:
                print(f"[fallback] trying {fallback}")
                ok = run(
                    pip_cmd("install", "--pre", "torch", "torchvision", "torchaudio", "--index-url", CUDA_INDEXES[fallback]),
                    dry_run=args.dry_run,
                    allow_failure=True,
                )
                if ok:
                    break
            if not ok:
                raise SystemExit("Could not install a CUDA PyTorch build. Try a different --cuda option or install Python 3.11/3.12.")

    print("\n[verify] PyTorch installation")
    verify_torch(dry_run=args.dry_run)

    if args.device == "gpu" and not args.dry_run:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit(
                "GPU setup finished, but torch.cuda.is_available() is still False. "
                "Check NVIDIA driver, Python version, and selected CUDA wheel index."
            )

    print("\nEnvironment setup finished.")


if __name__ == "__main__":
    main()
