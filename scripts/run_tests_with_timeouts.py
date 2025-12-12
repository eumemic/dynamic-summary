#!/usr/bin/env python3
"""Run pytest in bounded chunks with per-test time limits.

Why: Some environments enforce a wall-clock limit on a single command. This
wrapper enumerates tests and runs them in manageable chunks, while also
propagating a strict per-test timeout so no single test can hang the run.

Usage:
  python scripts/run_tests_with_timeouts.py \
    [--per-test-seconds 1.0] [--include-integration]

Recognized environment variables:
  RZ_MAX_TEST_DURATION: default seconds for --per-test-seconds (default: 1.0)

Exit codes:
  0 on success; 1 if any chunk fails or collection fails.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shlex
import signal
import subprocess
import sys
import time


def collect_tests(include_integration: bool) -> list[str]:
    marker = "not benchmark"
    if not include_integration:
        marker += " and not integration"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-m",
        marker,
    ]
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        print(f"[Runner] Failed to collect tests: {e}", file=sys.stderr)
        return []

    if res.returncode != 0:
        # Still attempt to extract nodeids from stdout if present
        print(res.stderr, file=sys.stderr)

    nodeids: list[str] = []
    for line in res.stdout.splitlines():
        # pytest -q --collect-only prints nodeids as lines (excluding collection headers)
        if not line or line.startswith("<"):
            continue
        # Heuristic: a nodeid contains '::' or ends with '.py'
        if "::" in line or line.endswith(".py"):
            nodeids.append(line.strip())
    return nodeids


def run_chunk(
    nodeids: list[str], per_test_seconds: float, include_integration: bool
) -> int:
    marker = "not benchmark"
    if not include_integration:
        marker += " and not integration"

    # Build command; run with xdist to keep runtime short
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-n",
        os.environ.get("PYTEST_XDIST_WORKERS", "8"),
        "--tb=short",
        "-m",
        marker,
        "--max-test-duration",
        str(per_test_seconds),
    ] + nodeids

    print(f"[Runner] pytest {' '.join(shlex.quote(p) for p in cmd[3:])}")
    res = subprocess.run(cmd, text=True)
    return res.returncode


def _kill_process_group(proc: subprocess.Popen) -> None:
    try:
        if hasattr(os, "getpgid"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            # Small grace, then KILL
            time.sleep(0.05)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.terminate()
            time.sleep(0.05)
            proc.kill()
    except Exception:
        pass


def run_per_node(
    nodeids: list[str], per_test_seconds: float, include_integration: bool
) -> tuple[int, int, int]:
    marker = "not benchmark"
    if not include_integration:
        marker += " and not integration"

    passed = 0
    failed = 0
    rc = 0

    for nid in nodeids:
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "-m",
            marker,
            "--max-test-duration",
            str(per_test_seconds),
            nid,
        ]
        print(f"[Runner] pytest {' '.join(shlex.quote(p) for p in cmd[3:])}")

        # Start in its own process group so we can kill the whole tree
        if os.name == "posix":
            preexec = os.setsid
        else:
            preexec = None
        start = time.time()
        proc = subprocess.Popen(cmd, preexec_fn=preexec, text=True)
        try:
            proc.wait(timeout=per_test_seconds + 0.2)
        except subprocess.TimeoutExpired:
            print(
                f"[Runner] ❌ Timeout: {nid} exceeded {per_test_seconds:.3f}s, killing",
                file=sys.stderr,
            )
            _kill_process_group(proc)
            failed += 1
            rc = 1
            continue
        duration = time.time() - start
        if proc.returncode == 0:
            passed += 1
        else:
            failed += 1
            rc = 1
        # Defensive guard: if a test took close to the limit but returned nonzero, enforce consistency
        if duration > per_test_seconds + 0.5:
            rc = 1
    return rc, passed, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-test-seconds",
        type=float,
        default=float(os.environ.get("RZ_MAX_TEST_DURATION", "2.0")),
        help="Hard per-test timeout in seconds (default from RZ_MAX_TEST_DURATION or 2.0)",
    )
    parser.add_argument(
        "--include-integration",
        action="store_true",
        help="Include integration tests (benchmarks always excluded)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.environ.get("RZ_TEST_CHUNK_SIZE", "200")),
        help="Number of tests to run per pytest invocation",
    )
    parser.add_argument(
        "--per-node",
        action="store_true",
        help="Run each test nodeid in a separate pytest process and kill after timeout",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=float(os.environ.get("RZ_MAX_WALL_DURATION", "0")),
        help="Optional overall wall-clock budget; exits early when exceeded",
    )
    parser.add_argument(
        "nodeids",
        nargs="*",
        help="Optional explicit test nodeids to run (skips collection when provided)",
    )
    args = parser.parse_args()

    if args.nodeids:
        nodeids = args.nodeids
    else:
        nodeids = collect_tests(include_integration=args.include_integration)
    if not nodeids:
        print("[Runner] No tests collected", file=sys.stderr)
        return 1

    start = time.time()

    if args.per_node:
        rc_total = 0
        passed_total = 0
        failed_total = 0
        for idx, nid in enumerate(nodeids, 1):
            if args.max_wall_seconds and (time.time() - start) >= args.max_wall_seconds:
                print(
                    f"[Runner] Reached wall-clock budget after {idx-1} tests; stopping early",
                    file=sys.stderr,
                )
                break
            rc, p, f = run_per_node(
                [nid], args.per_test_seconds, args.include_integration
            )
            rc_total = rc_total or rc
            passed_total += p
            failed_total += f
        total = passed_total + failed_total
        print(
            f"[Runner] Summary: {passed_total} passed, {failed_total} failed, total {total}"
        )
        return rc_total

    rc = 0
    for i in range(0, len(nodeids), args.chunk_size):
        if args.max_wall_seconds and (time.time() - start) >= args.max_wall_seconds:
            print(
                f"[Runner] Reached wall-clock budget at test {i}; stopping early",
                file=sys.stderr,
            )
            break
        chunk = nodeids[i : i + args.chunk_size]
        print(f"[Runner] Running tests {i + 1}..{i + len(chunk)} of {len(nodeids)}")
        r = run_chunk(chunk, args.per_test_seconds, args.include_integration)
        if r != 0:
            rc = r
            # Continue running remaining chunks to surface all failures
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
