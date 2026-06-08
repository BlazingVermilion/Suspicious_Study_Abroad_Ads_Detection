#!/usr/bin/env python3
"""Path helpers for scripts in this repository."""
from __future__ import annotations

import os
from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """Resolve the repository root. PROJECT_ROOT env var has priority."""
    if os.getenv("PROJECT_ROOT"):
        return Path(os.environ["PROJECT_ROOT"]).expanduser().resolve()

    current = (start or Path(__file__)).resolve()
    for parent in [current, *current.parents]:
        if (parent / "README.md").exists() and (parent / "src").exists():
            return parent
    return Path.cwd().resolve()


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
