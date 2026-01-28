#!/bin/bash
# Deploy ragzoom-memory plugin from source to ~/.claude/plugins/
#
# Usage: ./deploy-plugin.sh
#
# Creates timestamped backup of existing plugin, then syncs from source.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/../plugin"
TARGET_DIR="$HOME/.claude/plugins/ragzoom-memory"
BACKUP_BASE="$HOME/.claude/plugins/.ragzoom-memory-backups"

# Validate source exists
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: Source directory not found: $SOURCE_DIR" >&2
    exit 1
fi

if [[ ! -f "$SOURCE_DIR/.claude-plugin/plugin.json" ]]; then
    echo "Error: Not a valid plugin (missing .claude-plugin/plugin.json)" >&2
    exit 1
fi

# Create backup if target exists
if [[ -d "$TARGET_DIR" ]]; then
    mkdir -p "$BACKUP_BASE"
    backup_name="backup-$(date +%Y%m%d-%H%M%S)"
    backup_path="$BACKUP_BASE/$backup_name"

    echo "Backing up existing plugin to: $backup_path"
    cp -a "$TARGET_DIR" "$backup_path"

    # Keep only last 5 backups
    cd "$BACKUP_BASE"
    ls -t | tail -n +6 | xargs -I {} rm -rf "{}" 2>/dev/null || true
fi

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Sync with rsync (preserves permissions, deletes removed files)
echo "Deploying plugin..."
rsync -av --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    "$SOURCE_DIR/" "$TARGET_DIR/"

echo ""
echo "Deployed to: $TARGET_DIR"
echo ""

# Show what was deployed
echo "Contents:"
find "$TARGET_DIR" -type f | sed "s|$TARGET_DIR/||" | sort | head -20

file_count=$(find "$TARGET_DIR" -type f | wc -l | tr -d ' ')
if [[ "$file_count" -gt 20 ]]; then
    echo "... and $((file_count - 20)) more files"
fi

echo ""
echo "Done. Restart Claude Code to pick up changes."
