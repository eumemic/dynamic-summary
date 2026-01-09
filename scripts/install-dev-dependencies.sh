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

# Track packages to install
APT_PACKAGES=()

# Check for wget (needed to set up gh CLI repo)
if ! command -v wget &> /dev/null; then
    APT_PACKAGES+=(wget)
fi

# Check for GNU time (needed by run-checks.sh)
if [[ ! -x /usr/bin/time ]]; then
    APT_PACKAGES+=(time)
fi

# Check for gh CLI - need to add repo source first
GH_NEEDS_INSTALL=false
if ! command -v gh &> /dev/null; then
    GH_NEEDS_INSTALL=true
    APT_PACKAGES+=(gh)
fi

# Install apt packages if needed (single apt-get update)
if [[ ${#APT_PACKAGES[@]} -gt 0 ]]; then
    # If gh needs install and repo not yet added, set it up first
    if [[ "$GH_NEEDS_INSTALL" == "true" ]] && [[ ! -f /etc/apt/sources.list.d/github-cli.list ]]; then
        echo "Setting up GitHub CLI repository..."
        # Install wget first if needed (before adding gh repo)
        if ! command -v wget &> /dev/null; then
            apt-get update
            apt-get install -y wget
            # Remove wget from packages list since we just installed it
            APT_PACKAGES=("${APT_PACKAGES[@]/wget/}")
        fi
        mkdir -p -m 755 /etc/apt/keyrings
        out=$(mktemp)
        wget -nv -O"$out" https://cli.github.com/packages/githubcli-archive-keyring.gpg
        cat "$out" | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null
        chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    fi

    # Filter out empty elements and install remaining packages
    FILTERED_PACKAGES=()
    for pkg in "${APT_PACKAGES[@]}"; do
        [[ -n "$pkg" ]] && FILTERED_PACKAGES+=("$pkg")
    done

    if [[ ${#FILTERED_PACKAGES[@]} -gt 0 ]]; then
        echo "Installing system packages: ${FILTERED_PACKAGES[*]}"
        apt-get update
        apt-get install -y "${FILTERED_PACKAGES[@]}"
    fi
fi

# Install Python dependencies if not already installed
if ! python -c "import ragzoom" &> /dev/null; then
    echo "Installing Python dependencies from lockfile..."
    cd "$GIT_ROOT"
    pip install --upgrade "pip<24.1"
    # Use --ignore-installed to avoid conflicts with system packages
    # dev.lock includes numpy which is needed by type checking in post-tool-use hook
    pip install --ignore-installed -r requirements/dev.lock
elif ! python -c "import numpy" &> /dev/null; then
    # Edge case: ragzoom installed but numpy missing (shouldn't happen with lockfile)
    echo "Installing numpy..."
    pip install numpy
fi

echo "✅ Development dependencies installed"
