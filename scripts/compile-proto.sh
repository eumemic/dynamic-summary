#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="$ROOT_DIR/proto"
OUT_DIR="$ROOT_DIR/ragzoom/rpc"
PROTO_FILE="$PROTO_DIR/dynamic_summary.proto"

if ! python -c "import grpc_tools" >/dev/null 2>&1; then
  echo "❌ grpcio-tools is required. Install with: pip install grpcio-tools" >&2
  exit 1
fi

python -m grpc_tools.protoc \
  -I"$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$PROTO_FILE"

export RZ_REPO_ROOT="$ROOT_DIR"

python <<'PY'
from pathlib import Path
import os

ROOT = Path(os.environ["RZ_REPO_ROOT"])
OUT = ROOT / "ragzoom" / "rpc"
HEADERS = {
    "dynamic_summary_pb2.py": "# -*- coding: utf-8 -*-\n# mypy: ignore-errors\n# ruff: noqa\n# jscpd:ignore-start\n",
    "dynamic_summary_pb2_grpc.py": "# mypy: ignore-errors\n# ruff: noqa\n# jscpd:ignore-start\n",
}

for name, header in HEADERS.items():
    path = OUT / name
    text = path.read_text()
    header_lines = header.rstrip("\n").splitlines()
    lines = text.splitlines()
    if lines[: len(header_lines)] == header_lines:
        body_lines = lines[len(header_lines):]
    else:
        body_lines = lines
    while body_lines[: len(header_lines)] == header_lines:
        body_lines = body_lines[len(header_lines):]
    body = "\n".join(body_lines).lstrip("\n")
    text = header + body
    if name == "dynamic_summary_pb2_grpc.py":
        sentinel = "import dynamic_summary_pb2 as dynamic__summary__pb2"
        fallback = (
            "try:  # pragma: no cover\n"
            "    import dynamic_summary_pb2 as dynamic__summary__pb2\n"
            "except ModuleNotFoundError:  # pragma: no cover\n"
            "    from . import dynamic_summary_pb2 as dynamic__summary__pb2\n"
        )
        if sentinel in text and fallback.strip() not in text:
            text = text.replace(sentinel, fallback, 1)
    if "# jscpd:ignore-end" not in text:
        suffix = "\n# jscpd:ignore-end\n" if name.endswith("_grpc.py") else "\n\n# jscpd:ignore-end\n"
        text = text.rstrip() + suffix
    else:
        text = text.rstrip() + "\n"
    path.write_text(text)
PY

echo "✅ Generated protobuf sources in ragzoom/rpc"
