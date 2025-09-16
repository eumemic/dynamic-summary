#!/usr/bin/env python3
"""Find impacted tests for changed Python files.

Given a list of changed files, compute which tests are downstream:
- Include changed tests directly.
- For changed ragzoom modules, parse internal import graph (ragzoom.*) and find
  transitive dependents. Then include tests that import any impacted module.

Outputs a whitespace-separated list of test paths suitable for pytest.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]


def module_name_from_path(p: Path) -> str | None:
    try:
        rel = p.relative_to(ROOT)
    except ValueError:
        return None
    if not str(rel).endswith(".py"):
        return None
    parts = list(rel.parts)
    if parts[0] == "ragzoom":
        parts[-1] = parts[-1][:-3]
        return ".".join(parts)
    return None


def parse_imports(py_path: Path) -> Set[str]:
    src = py_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(py_path))
    except Exception:
        return set()
    mods: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith("ragzoom"):
                    mods.add(name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("ragzoom"):
                mods.add(node.module)
    return mods


def build_internal_graph() -> Tuple[Dict[str, Set[str]], Dict[str, Path]]:
    """Return (reverse_deps, module_to_path) for ragzoom.* modules."""
    mod_to_path: Dict[str, Path] = {}
    for p in ROOT.glob("ragzoom/**/*.py"):
        if p.name == "__init__.py":
            continue
        mn = module_name_from_path(p)
        if mn:
            mod_to_path[mn] = p

    # forward deps: mod -> set(imported mods)
    fwd: Dict[str, Set[str]] = {m: set() for m in mod_to_path}
    for m, p in mod_to_path.items():
        for dep in parse_imports(p):
            if dep in fwd:
                fwd[m].add(dep)
    # reverse deps: dep -> set(importers)
    rev: Dict[str, Set[str]] = {m: set() for m in mod_to_path}
    for m, deps in fwd.items():
        for d in deps:
            rev[d].add(m)
    return rev, mod_to_path


def transitive_dependents(changed: Set[str], rev: Dict[str, Set[str]]) -> Set[str]:
    out: Set[str] = set()
    stack = list(changed)
    while stack:
        m = stack.pop()
        if m in out:
            continue
        out.add(m)
        for imp in rev.get(m, ()):  # modules that import m
            if imp not in out:
                stack.append(imp)
    return out


def impacted_tests(changed_files: List[str]) -> List[str]:
    changed_paths = [Path(p).resolve() for p in changed_files if p.endswith(".py")]

    # Always include changed tests directly
    tests: Set[str] = set()
    for p in changed_paths:
        if "/tests/" in str(p):
            if p.name.startswith("test_"):
                tests.add(str(p))
            else:
                # Fixture/helpers (conftest, utils, __init__, etc.) may impact many tests
                tests.add(str(ROOT / "tests"))

    # Determine changed ragzoom modules
    changed_modules = {mn for p in changed_paths if (mn := module_name_from_path(p))}
    if not changed_modules:
        return sorted(tests)

    rev, _ = build_internal_graph()
    impacted_mods = transitive_dependents(changed_modules, rev)

    # Map tests -> imported ragzoom modules
    for tfile in ROOT.glob("tests/**/*.py"):
        try:
            mods = parse_imports(tfile)
        except Exception:
            mods = set()
        if mods & impacted_mods:
            tests.add(str(tfile))
    return sorted(tests)


def main() -> None:
    if len(sys.argv) < 2:
        print("")
        return
    files = sys.argv[1:]
    tests = impacted_tests(files)
    if tests:
        print(" ".join(tests))


if __name__ == "__main__":
    main()
