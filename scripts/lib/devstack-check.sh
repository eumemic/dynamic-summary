#!/bin/bash
# Shared helper for validating docker-compose stack is running
#
# Source this file, then call:
#   require_devstack_running  - Exit with error if stack not running
#   get_local_grpc_address    - Returns localhost:PORT
#   get_local_database_url    - Returns postgres connection string

devstack_grpc_running() {
    # Check if the grpc container is running
    # Uses docker compose ps with JSON output for reliable parsing
    docker compose ps --format json 2>/dev/null | \
        python3 -c "
import sys, json
for line in sys.stdin:
    c = json.loads(line)
    if c.get('Service') == 'grpc' and c.get('State') == 'running':
        sys.exit(0)
sys.exit(1)
" 2>/dev/null
}

require_devstack_running() {
    if ! devstack_grpc_running; then
        cat >&2 <<'ERROR'
Error: Local devstack is not running.

Start the docker-compose stack first:
  ./scripts/devstack start

Then retry your command.
ERROR
        exit 1
    fi
}

get_local_grpc_address() {
    # Returns localhost:PORT based on RAGZOOM_GRPC_PORT env var or default
    echo "localhost:${RAGZOOM_GRPC_PORT:-50051}"
}

get_local_database_url() {
    # Returns the host-accessible database URL
    # Port 5433 is exposed by docker-compose (maps to 5432 inside container)
    echo "postgresql://ragzoom:ragzoom@localhost:5433/ragzoom"
}
