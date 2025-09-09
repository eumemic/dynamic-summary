#!/usr/bin/env python3
"""Remove @pytest.mark.slow decorators from test files.

Usage:
  scripts/remove-slow-markers.py [PATH ...]

If no PATHs are provided, processes all Python files under tests/.
The script edits files in-place, removing lines that contain
`@pytest.mark.slow` or `pytestmark = pytest.mark.slow`.
"""

from __future__ import annotations

import sys
from pathlib import Path


def iter_targets(args: list[str]) -> list[Path]:
    if args:
        paths: list[Path] = []
        for a in args:
            p = Path(a)
            if p.is_dir():
                paths.extend(p.rglob("*.py"))
            elif p.suffix == ".py":
                paths.append(p)
        return list({p.resolve() for p in paths})
    else:
        return [p for p in Path("tests").rglob("*.py")]


def process_file(path: Path) -> bool:
    try:
        original = path.read_text(encoding="utf-8")
    except Exception:
        return False
    lines = original.splitlines()
    changed = False
    filtered: list[str] = []
    for line in lines:
        lstripped = line.lstrip()
        if (
            lstripped.startswith("@pytest.mark.slow")
            or "pytestmark = pytest.mark.slow" in lstripped
        ):
            changed = True
            continue
        filtered.append(line)
    if changed:
        path.write_text("\n".join(filtered) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    targets = iter_targets(sys.argv[1:])
    changed = 0
    for p in targets:
        if process_file(p):
            changed += 1
            print(f"[removed] {p}")
    print(f"Removed slow markers from {changed} files.")


if __name__ == "__main__":
    main()

