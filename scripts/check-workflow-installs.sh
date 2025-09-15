#!/usr/bin/env bash
set -euo pipefail

dir=".github/workflows"
if [ ! -d "$dir" ]; then
  echo "[WorkflowPins] ✅ No workflows directory; nothing to check"
  exit 0
fi

# Collect all pip install occurrences
if command -v rg >/dev/null 2>&1; then
  hits=$(rg -n "pip install|pip-sync" "$dir" || true)
else
  hits=$(grep -RinE "pip install|pip-sync" "$dir" || true)
fi

# Filter out allowed patterns
# Allowed:
# - pip install --upgrade pip
# - pip install pip-tools
# - pip-sync requirements/*.lock
# - pip install pytest-cov
# - pip install awscli
# - pip install -e .[...]
filtered=$(echo "$hits" | grep -Ev \
  -e "pip install --upgrade pip" \
  -e "pip install pip-tools" \
  -e "pip-sync( |$)" \
  -e "pip install (.+/)?requirements/.+\.lock" \
  -e "pip install pytest-cov" \
  -e "pip install awscli" \
  -e "pip install -e \\..*\\[" \
  || true)

# Also ignore empty output
if [ -z "${filtered// }" ]; then
  echo "[WorkflowPins] ✅ No disallowed pip installs found in workflows"
  exit 0
fi

echo "[WorkflowPins] ❌ Disallowed pip install lines detected in workflows:" >&2
echo "$filtered" >&2
exit 1

