---
status: COMPLETE
---

# gRPC CLI Architecture

## Overview

All ragzoom CLI commands that need server data must work through gRPC to support remote server operation. The server may run on a different machine from the CLI client.

## Goals

1. **Remote-capable commands** - Commands needing server data work via gRPC
2. **Fail fast** - CLI fails immediately with clear error if server unreachable
3. **No auto-start** - User manages server lifecycle, CLI doesn't spawn daemons

## Non-Goals

- Auto-start daemon (explicitly removed - see daemon-lifecycle.md history)
- Local fallback for gRPC commands
- Migrating legitimate local-only commands (inspect, export, eval)

## Command Classification

### Already Using gRPC (no changes needed)

- `index` / `append` - IndexDocument, AppendText, BatchAppendText
- `query` - ExecuteQuery
- `document-status` - GetDocumentStatus
- `clear` - ClearDocument
- `telemetry` / `telemetry-export` - GetTelemetry, ExportTelemetry

### Requiring Migration to gRPC

| Command | Current Implementation | New gRPC Method |
|---------|----------------------|-----------------|
| `documents` | DocumentService.list_documents() | ListDocuments |
| `status` | DocumentService.get_system_status() | GetSystemStatus |
| `cost` | node_repo.get_cost_stats() | GetCostStats |

Note: `pin` command is obsolete and will be removed (not migrated to gRPC).

### Staying Local (legitimate client-side work)

| Command | Reason |
|---------|--------|
| `validate` | Needs `--complete` and `--telemetry-file` options for benchmark compatibility |
| `inspect` | Debugging tool, needs full node details including embeddings |
| `export` | Large data export, client-side file writing |
| `eval measure/report/compare` | Client-side computation on sampled data |
| `serve` | Starts REST server (is a server, not client) |
| `server *` | Daemon management commands |
| `config` | Local configuration display |

**Important limitation:** Local commands require direct database access. They will NOT work when the server is on a remote machine. This is acceptable because:
- `inspect`/`export` are debugging tools typically used locally
- `eval` processes data locally anyway
- Future work could add gRPC endpoints if remote use cases emerge

## New gRPC Methods

Add to WorkerService in `proto/dynamic_summary.proto`:

```protobuf
// List all indexed documents
rpc ListDocuments(ListDocumentsRequest) returns (ListDocumentsResponse);

message ListDocumentsRequest {}
message ListDocumentsResponse {
  repeated DocumentInfo documents = 1;
}
message DocumentInfo {
  string document_id = 1;
  int64 leaf_count = 2;
  int64 node_count = 3;
  bool is_temporal = 4;
  optional string time_start = 5;  // ISO 8601
  optional string time_end = 6;    // ISO 8601
  optional double completion_pct = 7;
}

// Get system-wide status (matches existing DocumentService.get_system_status())
rpc GetSystemStatus(GetSystemStatusRequest) returns (GetSystemStatusResponse);

message GetSystemStatusRequest {}
message GetSystemStatusResponse {
  int64 total_nodes = 1;
  int64 leaf_nodes = 2;
  int64 tree_depth = 3;
}

// Get cost statistics
rpc GetCostStats(GetCostStatsRequest) returns (GetCostStatsResponse);

message GetCostStatsRequest {
  optional string document_id = 1;  // If omitted, returns all docs
}
message GetCostStatsResponse {
  repeated DocumentCostStats documents = 1;
}
message DocumentCostStats {
  string document_id = 1;
  double total_cost = 2;       // Sum of all node costs (arbitrary units)
  int64 total_nodes = 3;       // Total node count
  int64 leaf_nodes = 4;        // Leaf node count
  int64 summary_nodes = 5;     // Summary node count (total - leaf)
}

```

### Error Handling

All new methods follow existing patterns:
- **NOT_FOUND**: Document or node doesn't exist
- **INVALID_ARGUMENT**: Bad input (empty document_id, etc.)
- **INTERNAL**: Unexpected server error

## Auto-Start Removal

### Current State (Partially Done)

There's an uncommitted change in `ragzoom/cli.py` that:
- Renamed `_resolve_server_address_with_autostart` → `_resolve_server_address`
- Removed the `ensure_server_running()` call
- **But didn't update the 6 call sites** that still reference the old function name

This spec completes that work by:
1. Updating all call sites to use `_resolve_server_address`
2. Removing the unused `ensure_server_running` import

### Old Behavior (being removed)

```python
def _resolve_server_address_with_autostart(value: str | None) -> str:
    if value:
        return value
    ensure_server_running()  # REMOVE THIS
    return f"localhost:{PRODUCTION_PORT}"
```

### New Behavior

```python
def _resolve_server_address(value: str | None) -> str:
    """Resolve server address, fail fast if not reachable."""
    address = value or f"localhost:{PRODUCTION_PORT}"

    # Quick TCP connectivity check
    host, port_str = address.rsplit(":", 1)
    port = int(port_str)
    try:
        with socket.create_connection((host, port), timeout=2):
            pass  # Connection succeeded
    except (OSError, TimeoutError):
        raise click.ClickException(
            f"Cannot connect to RagZoom server at {address}.\n"
            f"Start the server with: ragzoom server start"
        )

    return address
```

This uses a simple TCP socket check rather than a full gRPC health check - it's faster and sufficient to detect "server not running".

### Error Message

When server is unreachable:

```
Error: Cannot connect to RagZoom server at localhost:50051.
Start the server with: ragzoom server start

For remote servers, specify the address:
  ragzoom <command> --server hostname:port
```

## CLI Changes

### Shared Server Option

All gRPC commands use a common option:

```python
@click.option(
    "--server", "-s",
    default=None,
    help="Server address (host:port). Default: localhost:50051"
)
```

### Commands Using This Option

- `index`, `query`, `document-status`, `clear`, `telemetry`, `telemetry-export` (existing)
- `documents`, `validate`, `status`, `cost` (migrated)

## Implementation

### Files to Modify

1. **proto/dynamic_summary.proto** - Add new RPC methods and messages
2. **ragzoom/rpc/** - Regenerate Python stubs after proto changes
3. **ragzoom/server/servicers.py** - Implement new servicer methods
4. **ragzoom/grpc_client.py** - Add client methods for new RPCs
5. **ragzoom/cli.py** - Migrate commands, remove auto-start

### Migration Steps

1. Add proto definitions, regenerate stubs
2. Implement servicers (server-side)
3. Add client methods
4. Migrate CLI commands one by one
5. Remove `ensure_server_running()` and auto-start logic
6. Remove unused direct-DB imports from CLI

## Testing

### Unit Tests

- Each new servicer method has tests
- CLI commands test server-unreachable error path
- Verify no auto-start attempts

### Integration Tests

- All migrated commands work against running server
- Commands fail fast with clear error when server is down
- Remote server scenario (different host:port)

### Manual Verification

```bash
# Ensure server is stopped
ragzoom server stop

# Verify fail-fast behavior
ragzoom documents  # Should fail immediately with clear error

# Start server
ragzoom server start --daemon

# Verify migrated commands work
ragzoom documents
ragzoom status
ragzoom cost

# Test remote scenario
ragzoom documents --server otherhost:50051

# Local commands still work without server
ragzoom validate mydoc
ragzoom inspect abc123
ragzoom export -d mydoc
ragzoom eval measure -d mydoc
```

## Deprecations

### Pin Command Removal

The `pin` command and all pinned node functionality is obsolete and will be removed:
- Remove `ragzoom pin` CLI command
- Remove `pin_node()` from DocumentService
- Remove `pinned_nodes` from SystemStatus
- Remove `pinned` column from nodes table (future migration)

This is a separate cleanup task, not blocking gRPC migration.

## Relationship to Other Specs

- **daemon-lifecycle.md**: That spec originally described auto-start as a goal. Auto-start is now explicitly NOT implemented. User is responsible for starting the server.
