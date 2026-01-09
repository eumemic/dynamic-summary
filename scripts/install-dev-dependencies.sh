#!/bin/bash
# Install development dependencies for RagZoom
#
# This script is idempotent - safe to run multiple times.
# Used by the session-start hook for Claude Code on the web.

set -euo pipefail

# Get repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Installing development dependencies..."

# Track if we need apt-get update
NEED_APT_UPDATE=false
APT_PACKAGES=()

# Check for gh CLI
if ! command -v gh &> /dev/null; then
    echo "Setting up GitHub CLI repository..."
    (type -p wget >/dev/null || (apt-get update && apt-get install wget -y)) \
    && mkdir -p -m 755 /etc/apt/keyrings \
    && out=$(mktemp) && wget -nv -O"$out" https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    && cat "$out" | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    NEED_APT_UPDATE=true
    APT_PACKAGES+=(gh)
fi

# Check for GNU time (needed by run-checks.sh)
if [[ ! -x /usr/bin/time ]]; then
    NEED_APT_UPDATE=true
    APT_PACKAGES+=(time)
fi

# Install apt packages if needed
if [[ "$NEED_APT_UPDATE" == "true" ]] && [[ ${#APT_PACKAGES[@]} -gt 0 ]]; then
    echo "Installing system packages: ${APT_PACKAGES[*]}"
    apt-get update
    apt-get install -y "${APT_PACKAGES[@]}"
fi

# Install Python dependencies if not already installed
# numpy is installed first as it's needed by type checking in post-tool-use hook
if ! python -c "import numpy" &> /dev/null; then
    echo "Installing numpy..."
    pip install numpy
fi

if ! python -c "import ragzoom" &> /dev/null; then
    echo "Installing Python dependencies from lockfile..."
    cd "$GIT_ROOT"
    pip install --upgrade "pip<24.1"
    # Use --ignore-installed to avoid conflicts with system packages
    pip install --ignore-installed -r requirements/dev.lock
fi

echo "✅ Development dependencies installed"
