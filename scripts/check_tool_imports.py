#!/usr/bin/env python3
"""Load every packaged tool without executing data-dependent main routines."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def main() -> None:
    sys.path.insert(0, str(TOOLS))
    paths = sorted(TOOLS.glob("*.py"))
    for path in paths:
        runpy.run_path(str(path), run_name="__reviewer_import_check__")
    print(f"FocalSwap tool import check passed: {len(paths)} modules")


if __name__ == "__main__":
    main()
