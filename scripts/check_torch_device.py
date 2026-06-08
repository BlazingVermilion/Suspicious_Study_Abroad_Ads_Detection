#!/usr/bin/env python3
"""Print the active PyTorch device configuration."""
from __future__ import annotations

try:
    import torch
except Exception as exc:
    raise SystemExit(f"Could not import torch: {exc}")

print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU / NO CUDA")
