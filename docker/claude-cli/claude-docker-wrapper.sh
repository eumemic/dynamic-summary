#!/usr/bin/env bash
# Wrapper script that the Claude Agent SDK calls as cli_path.
# Forwards the invocation into a running Docker container via `docker exec -i`.
#
# The SDK launches this script exactly as it would launch the `claude` binary,
# piping stdin/stdout for JSON-RPC communication.  `exec` at the end replaces
# this shell process so the pipes connect directly to the containerised CLI.

set -euo pipefail

CONTAINER="${RAGZOOM_CLAUDE_CONTAINER:-ragzoom-claude-cli}"

# Build -e flags for env vars the CLI needs inside the container.
ENV_FLAGS=()
for var in ANTHROPIC_API_KEY XDG_DATA_HOME \
           CLAUDE_CODE_ENTRYPOINT CLAUDE_AGENT_SDK_VERSION \
           CLAUDE_CODE_STREAM_CLOSE_TIMEOUT; do
    if [[ -n "${!var:-}" ]]; then
        ENV_FLAGS+=(-e "${var}=${!var}")
    fi
done

# Forward any CLAUDE_CODE_* vars not already covered above.
while IFS='=' read -r key value; do
    case "$key" in
        CLAUDE_CODE_ENTRYPOINT|CLAUDE_CODE_STREAM_CLOSE_TIMEOUT) ;;  # already handled
        CLAUDE_CODE_*) ENV_FLAGS+=(-e "${key}=${value}") ;;
    esac
done < <(env)

exec docker exec -i "${ENV_FLAGS[@]}" "$CONTAINER" claude "$@"
