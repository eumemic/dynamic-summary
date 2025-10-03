#!/usr/bin/env python3
"""Compatibility wrapper for ragzoom.tools.incremental_index."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ragzoom.tools.incremental_index import cli_main  # noqa: E402

if __name__ == "__main__":
    cli_main()
