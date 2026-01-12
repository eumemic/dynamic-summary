#!/bin/bash
# Install development dependencies for RagZoom
#
# This script is idempotent - safe to run multiple times.
# Used by the session-start hook for Claude Code on the web.
#
# In restricted environments (like Claude Code on the web), apt-get may fail
# due to permission issues. The script handles this gracefully - apt packages
# are nice-to-have but Python dependencies are the critical requirement.

set -uo pipefail  # Don't use -e, we handle errors explicitly

# Get repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Installing development dependencies..."

# Remove uv-isolated tools that conflict with pip-installed packages
# uv creates isolated Python environments for tools, which don't see system packages.
# This causes issues with mypy/pytest not finding numpy, sqlalchemy, etc.
remove_conflicting_uv_tools() {
    local tools_to_check=("mypy" "pytest" "ruff" "black")

    for tool in "${tools_to_check[@]}"; do
        local tool_path
        tool_path=$(command -v "$tool" 2>/dev/null) || continue

        # Check if it's a uv-isolated tool by examining shebang
        if [[ -f "$tool_path" ]] && head -1 "$tool_path" 2>/dev/null | grep -q "uv/tools"; then
            echo "Removing uv-isolated $tool (conflicts with pip packages)..."
            local uv_tool_dir="/root/.local/share/uv/tools/$tool"
            rm -rf "$uv_tool_dir" 2>/dev/null || true
            rm -f "/root/.local/bin/$tool" 2>/dev/null || true
            rm -f "/root/.local/bin/dm${tool}" 2>/dev/null || true  # For dmypy
        fi
    done
}

# Remove conflicting uv tools before installing
remove_conflicting_uv_tools

# Install gh CLI from pre-built binary (works in sandboxed environments where apt fails)
GH_VERSION="2.63.2"

install_gh_cli() {
    if command -v gh &> /dev/null; then
        echo "✓ gh CLI already installed"
        return 0
    fi

    echo "Installing gh CLI from binary..."

    local arch
    arch=$(uname -m)
    case "$arch" in
        x86_64) arch="amd64" ;;
        aarch64) arch="arm64" ;;
        *) echo "⚠ Unsupported architecture: $arch"; return 1 ;;
    esac

    local tarball="gh_${GH_VERSION}_linux_${arch}.tar.gz"
    local url="https://github.com/cli/cli/releases/download/v${GH_VERSION}/${tarball}"

    if curl -sL "$url" | tar xz -C /tmp && \
       mv "/tmp/gh_${GH_VERSION}_linux_${arch}/bin/gh" /usr/local/bin/; then
        rm -rf "/tmp/gh_${GH_VERSION}_linux_${arch}"
        echo "✓ gh CLI installed"
        return 0
    else
        echo "⚠ Could not install gh CLI"
        return 1
    fi
}

# Install gh CLI (non-fatal if it fails)
install_gh_cli || true

# Install Python dependencies - this is the critical part
install_python_deps() {
    # Check if core packages are importable
    if python -c "import ragzoom; import numpy; import pytest; import grpc; import mypy" &> /dev/null; then
        echo "✓ Python dependencies already installed"
        return 0
    fi

    echo "Installing Python dependencies from lockfile..."
    cd "$GIT_ROOT"

    # Upgrade pip first (avoid compatibility issues)
    pip install --quiet --upgrade "pip<24.1" || true

    # Install from lockfile - includes all dev dependencies
    # Use --ignore-installed to avoid conflicts with system packages
    if pip install --quiet --ignore-installed -r requirements/dev.lock; then
        echo "✓ Python dependencies installed"
        return 0
    else
        echo "❌ Failed to install Python dependencies"
        return 1
    fi
}

# Python deps are required - fail if they don't install
if ! install_python_deps; then
    echo "❌ Critical: Python dependencies failed to install"
    exit 1
fi

echo "✅ Development dependencies ready"
