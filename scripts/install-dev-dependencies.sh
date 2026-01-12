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

# Try to install apt packages, but don't fail if apt-get doesn't work
# (common in sandboxed/restricted environments)
install_apt_packages() {
    # Check for gh CLI
    if command -v gh &> /dev/null; then
        echo "✓ gh CLI already installed"
        return 0
    fi

    echo "Attempting to install system packages (gh CLI)..."

    # Try setting up gh repo and installing
    if [[ ! -f /etc/apt/sources.list.d/github-cli.list ]]; then
        # Ensure wget is available
        if ! command -v wget &> /dev/null; then
            if ! apt-get update -qq 2>/dev/null && apt-get install -y wget 2>/dev/null; then
                echo "⚠ Could not install wget (apt-get unavailable or restricted)"
                return 1
            fi
        fi

        # Set up gh CLI repo
        mkdir -p -m 755 /etc/apt/keyrings 2>/dev/null || true
        local keyring_file
        keyring_file=$(mktemp 2>/dev/null) || keyring_file="/tmp/gh-keyring-$$"
        if wget -nv -O"$keyring_file" https://cli.github.com/packages/githubcli-archive-keyring.gpg 2>/dev/null; then
            cat "$keyring_file" > /etc/apt/keyrings/githubcli-archive-keyring.gpg 2>/dev/null || true
            chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg 2>/dev/null || true
            echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list 2>/dev/null || true
        fi
        rm -f "$keyring_file" 2>/dev/null || true
    fi

    # Try to install gh
    if apt-get update -qq 2>/dev/null && apt-get install -y gh 2>/dev/null; then
        echo "✓ gh CLI installed"
        return 0
    else
        echo "⚠ Could not install gh CLI (apt-get unavailable or restricted)"
        echo "  This is optional - PR creation will require manual gh setup"
        return 1
    fi
}

# Attempt apt installations (non-fatal if they fail)
install_apt_packages || true

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
