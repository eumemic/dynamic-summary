# Implementation Plan: gRPC CLI Architecture

**Spec:** `specs/grpc-cli-architecture.md`
**Status:** READY

## Overview

Migrate CLI commands `documents`, `validate`, `status`, and `cost` from direct database access to gRPC, remove auto-start behavior, and clean up obsolete pin functionality.

---

## Phase 1: Proto Definitions

All new gRPC methods must be defined before server/client implementation.

### 1.1 ListDocuments RPC

- [x] Add ListDocuments RPC to WorkerService
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: `grpc_tools_protoc` compiles without errors, pb2 files contain ListDocuments
  - Test: tests/test_grpc_proto.py::TestListDocumentsProto
  - Location: proto/dynamic_summary.proto:318 (in WorkerService)

- [x] Add ListDocumentsRequest message (empty)
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: Message defined with correct structure
  - Test: tests/test_grpc_proto.py::TestListDocumentsProto::test_list_documents_request_exists
  - Location: proto/dynamic_summary.proto:299

- [x] Add ListDocumentsResponse message with repeated DocumentInfo
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: Message contains `repeated DocumentInfo documents = 1`
  - Test: tests/test_grpc_proto.py::TestListDocumentsProto::test_list_documents_response_has_repeated_documents
  - Location: proto/dynamic_summary.proto:301-303

- [x] Add DocumentInfo message
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: Fields: document_id, leaf_count, node_count, is_temporal, time_start, time_end, completion_pct
  - Test: tests/test_grpc_proto.py::TestListDocumentsProto::test_document_info_has_required_fields
  - Location: proto/dynamic_summary.proto:305-313

### 1.2 ValidateDocument RPC

- [x] Add ValidateDocument RPC to WorkerService
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: RPC defined accepting ValidateDocumentRequest, returning ValidateDocumentResponse
  - Test: tests/test_grpc_proto.py::TestValidateDocumentProto
  - Location: proto/dynamic_summary.proto:334

- [x] Add ValidateDocumentRequest message
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: Contains `string document_id = 1`
  - Test: tests/test_grpc_proto.py::TestValidateDocumentProto::test_validate_document_request_exists
  - Location: proto/dynamic_summary.proto:317-319

- [x] Add ValidateDocumentResponse message
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: Contains `bool valid = 1`, `repeated string errors = 2`
  - Test: tests/test_grpc_proto.py::TestValidateDocumentProto::test_validate_document_response_has_required_fields
  - Location: proto/dynamic_summary.proto:321-324

### 1.3 GetSystemStatus RPC

- [x] Add GetSystemStatus RPC to WorkerService
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: RPC defined accepting GetSystemStatusRequest, returning GetSystemStatusResponse
  - Test: tests/test_grpc_proto.py::TestGetSystemStatusProto
  - Location: proto/dynamic_summary.proto:345

- [x] Add GetSystemStatusRequest message (empty)
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: Empty message defined
  - Test: tests/test_grpc_proto.py::TestGetSystemStatusProto::test_get_system_status_request_exists
  - Location: proto/dynamic_summary.proto:328

- [x] Add GetSystemStatusResponse message
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: Contains total_nodes, leaf_nodes, tree_depth (no pinned_nodes)
  - Test: tests/test_grpc_proto.py::TestGetSystemStatusProto::test_get_system_status_response_has_required_fields
  - Location: proto/dynamic_summary.proto:330-334

### 1.4 GetCostStats RPC

- [x] Add GetCostStats RPC to WorkerService
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: RPC defined accepting GetCostStatsRequest, returning GetCostStatsResponse
  - Test: tests/test_grpc_proto.py::TestGetCostStatsProto
  - Location: proto/dynamic_summary.proto:357

- [x] Add GetCostStatsRequest message
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: Contains `optional string document_id = 1`
  - Test: tests/test_grpc_proto.py::TestGetCostStatsProto::test_get_cost_stats_request_exists
  - Location: proto/dynamic_summary.proto:337-339

- [x] Add GetCostStatsResponse message
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: Contains `repeated DocumentCostStats documents = 1`
  - Test: tests/test_grpc_proto.py::TestGetCostStatsProto::test_get_cost_stats_response_has_repeated_documents
  - Location: proto/dynamic_summary.proto:341-343

- [x] Add DocumentCostStats message
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: Contains document_id, total_cost, total_nodes, leaf_nodes, summary_nodes
  - Test: tests/test_grpc_proto.py::TestGetCostStatsProto::test_document_cost_stats_has_required_fields
  - Location: proto/dynamic_summary.proto:345-351

### 1.5 Regenerate Proto Stubs

- [x] Run proto compilation to regenerate Python stubs
  - Spec: specs/grpc-cli-architecture.md § Implementation
  - Success: `ragzoom/rpc/dynamic_summary_pb2.py` and `*_pb2_grpc.py` updated
  - Test: `python -c "from ragzoom.rpc import dynamic_summary_pb2 as pb; pb.ListDocumentsRequest()"`
  - Location: ragzoom/rpc/dynamic_summary_pb2.py (generated)

---

## Phase 2: Server Servicers

Implement server-side handlers for the new RPCs.

### 2.1 ListDocuments Servicer

- [x] Implement ListDocuments method in WorkerServicer
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: Returns DocumentInfo for all documents with correct fields
  - Test: tests/test_grpc_server_integration.py::test_list_documents_servicer_*
  - Location: ragzoom/server/servicers.py:1183-1236

### 2.2 ValidateDocument Servicer

- [x] Implement ValidateDocument method in WorkerServicer
  - Spec: specs/grpc-cli-architecture.md § Error Handling
  - Success: Returns valid=true when document passes, errors list when fails
  - Test: tests/test_grpc_server_integration.py::test_validate_document_servicer_returns_valid_for_healthy_document
  - Location: ragzoom/server/servicers.py:1238-1280

- [x] Handle NOT_FOUND for missing document
  - Spec: specs/grpc-cli-architecture.md § Error Handling
  - Success: Aborts with NOT_FOUND status code
  - Test: tests/test_grpc_server_integration.py::test_validate_document_servicer_not_found_for_missing_document
  - Location: ragzoom/server/servicers.py:1257-1262 (within ValidateDocument)

- [x] Handle INVALID_ARGUMENT for empty document_id
  - Spec: specs/grpc-cli-architecture.md § Error Handling
  - Success: Aborts with INVALID_ARGUMENT status code
  - Test: tests/test_grpc_server_integration.py::test_validate_document_servicer_invalid_argument_for_empty_document_id
  - Location: ragzoom/server/servicers.py:1250-1255 (within ValidateDocument)

### 2.3 GetSystemStatus Servicer

- [x] Implement GetSystemStatus method in WorkerServicer
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: Returns aggregated total_nodes, leaf_nodes, tree_depth across all documents
  - Test: test_get_system_status_servicer (new)
  - Location: ragzoom/server/servicers.py:1285-1314

### 2.4 GetCostStats Servicer

- [x] Implement GetCostStats method in WorkerServicer
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: Returns cost stats for specified document or all documents
  - Test: test_get_cost_stats_servicer_returns_stats_for_document, test_get_cost_stats_servicer_returns_all_documents
  - Location: ragzoom/server/servicers.py:1313-1358

- [x] Handle optional document_id filter
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: When document_id provided, returns only that doc; when omitted, returns all
  - Test: test_get_cost_stats_servicer_returns_all_documents (existing)
  - Location: ragzoom/server/servicers.py:1328-1360 (within GetCostStats)

---

## Phase 3: gRPC Client Methods

Add client-side wrappers for the new RPCs.

### 3.1 list_documents() Client Method

- [x] Add list_documents() method to GrpcRagzoomClient
  - Spec: specs/grpc-cli-architecture.md § CLI Changes
  - Success: Method returns list of DocumentInfoView dataclasses
  - Test: test_grpc_client_list_documents, test_grpc_client_list_documents_empty
  - Location: ragzoom/client/grpc_client.py:960-991

- [x] Add DocumentInfoView dataclass for client response
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: Dataclass with document_id, leaf_count, node_count, is_temporal, time_start, time_end, completion_pct
  - Test: test_grpc_client_list_documents
  - Location: ragzoom/client/grpc_client.py:242-264

### 3.2 validate_document() Client Method

- [x] Add validate_document() method to GrpcRagzoomClient
  - Spec: specs/grpc-cli-architecture.md § CLI Changes
  - Success: Returns ValidationResult with valid: bool, errors: list[str]
  - Test: test_grpc_client_validate_document, test_grpc_client_validate_document_not_found, test_grpc_client_validate_document_empty_id
  - Location: ragzoom/client/grpc_client.py:979-1003

- [x] Add ValidationResult dataclass for client response
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: Dataclass with valid: bool, errors: list[str]
  - Test: test_grpc_client_validate_document
  - Location: ragzoom/client/grpc_client.py:265-277

### 3.3 get_system_status() Client Method

- [x] Add get_system_status() method to GrpcRagzoomClient
  - Spec: specs/grpc-cli-architecture.md § CLI Changes
  - Success: Returns SystemStatusView with total_nodes, leaf_nodes, tree_depth
  - Test: test_grpc_client_get_system_status, test_grpc_client_get_system_status_empty
  - Location: ragzoom/client/grpc_client.py:1021-1038

- [x] Add SystemStatusView dataclass for client response
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: Dataclass with total_nodes, leaf_nodes, tree_depth (no pinned_nodes)
  - Test: test_grpc_client_get_system_status
  - Location: ragzoom/client/grpc_client.py:281-295

### 3.4 get_cost_stats() Client Method

- [x] Add get_cost_stats() method to GrpcRagzoomClient
  - Spec: specs/grpc-cli-architecture.md § CLI Changes
  - Success: Returns list of CostStatsView dataclasses
  - Test: test_grpc_client_get_cost_stats, test_grpc_client_get_cost_stats_all_documents, test_grpc_client_get_cost_stats_empty
  - Location: ragzoom/client/grpc_client.py:1080-1110

- [x] Add CostStatsView dataclass for client response
  - Spec: specs/grpc-cli-architecture.md § New gRPC Methods
  - Success: Dataclass with document_id, total_cost, total_nodes, leaf_nodes, summary_nodes
  - Test: test_grpc_client_get_cost_stats
  - Location: ragzoom/client/grpc_client.py:300-317

---

## Phase 4: CLI Command Migration

Migrate CLI commands from direct DB access to gRPC client calls.

### 4.1 Migrate `documents` Command

- [x] Replace DocumentService.list_documents() with gRPC client.list_documents()
  - Spec: specs/grpc-cli-architecture.md § Commands Requiring Migration
  - Success: `ragzoom documents` works against running server, fails fast if server down
  - Test: test_documents_uses_grpc, test_documents_empty_list, test_documents_temporal_document
  - Location: ragzoom/cli.py:575-612

- [x] Add --server-address option to `documents` command
  - Spec: specs/grpc-cli-architecture.md § Shared Server Option
  - Success: Option accepts host:port, defaults to localhost:50051
  - Test: test_documents_uses_grpc (uses fixture which mocks server address resolution)
  - Location: ragzoom/cli.py:576-582

### 4.2 Migrate `validate` Command

- [x] Replace validate_document() with gRPC client.validate_document()
  - Spec: specs/grpc-cli-architecture.md § Commands Requiring Migration
  - Success: `ragzoom validate doc` works against running server
  - Test: test_cli_validate_uses_grpc
  - Location: ragzoom/cli.py:740-772

- [x] Add --server-address option to `validate` command
  - Spec: specs/grpc-cli-architecture.md § Shared Server Option
  - Success: Option accepts host:port, defaults to localhost:50051
  - Test: test_cli_validate_server_option
  - Location: ragzoom/cli.py:743-748

- [x] Remove direct store/vector_index creation from validate
  - Spec: specs/grpc-cli-architecture.md § CLI Changes
  - Success: No create_store_with_docker or create_vector_index in validate command
  - Test: test_cli_validate_uses_grpc
  - Location: ragzoom/cli.py (removed `from ragzoom.validation import validate_document`)

### 4.3 Migrate `status` Command

- [ ] Replace DocumentService.get_system_status() with gRPC client.get_system_status()
  - Spec: specs/grpc-cli-architecture.md § Commands Requiring Migration
  - Success: `ragzoom status` works against running server
  - Test: test_cli_status_uses_grpc (new)
  - Location: ragzoom/cli.py:1163-1191

- [ ] Add --server-address option to `status` command
  - Spec: specs/grpc-cli-architecture.md § Shared Server Option
  - Success: Option accepts host:port, defaults to localhost:50051
  - Test: test_cli_status_server_option (new)
  - Location: ragzoom/cli.py:1161-1163

- [ ] Remove pinned_nodes display from status output
  - Spec: specs/grpc-cli-architecture.md § Pin Command Removal
  - Success: Status output no longer shows "Pinned nodes: X"
  - Test: test_cli_status_no_pinned_nodes (new)
  - Location: ragzoom/cli.py:1183

### 4.4 Migrate `cost` Command

- [ ] Replace node_repo.get_cost_stats() with gRPC client.get_cost_stats()
  - Spec: specs/grpc-cli-architecture.md § Commands Requiring Migration
  - Success: `ragzoom cost doc` works against running server
  - Test: test_cli_cost_uses_grpc (new)
  - Location: ragzoom/cli.py:1197-1251

- [ ] Add --server-address option to `cost` command
  - Spec: specs/grpc-cli-architecture.md § Shared Server Option
  - Success: Option accepts host:port, defaults to localhost:50051
  - Test: test_cli_cost_server_option (new)
  - Location: ragzoom/cli.py:1195-1197

---

## Phase 5: Auto-Start Removal

Remove daemon auto-start behavior from CLI.

### 5.1 Replace Auto-Start Function

- [ ] Rename _resolve_server_address_with_autostart to _resolve_server_address
  - Spec: specs/grpc-cli-architecture.md § Auto-Start Removal
  - Success: Function renamed in cli.py
  - Test: test_no_autostart_function_exists (new)
  - Location: ragzoom/cli.py:130-143

- [ ] Replace auto-start with TCP connectivity check
  - Spec: specs/grpc-cli-architecture.md § New Behavior
  - Success: Function checks TCP connectivity, raises ClickException if unreachable
  - Test: test_resolve_server_address_fails_fast (new)
  - Location: ragzoom/cli.py:130-143

- [ ] Show helpful error message with server start command
  - Spec: specs/grpc-cli-architecture.md § Error Message
  - Success: Error includes "Start the server with: ragzoom server start"
  - Test: test_server_unreachable_error_message (new)
  - Location: ragzoom/cli.py:140-143

### 5.2 Update Call Sites

- [ ] Update call site at line 385 (telemetry command)
  - Spec: specs/grpc-cli-architecture.md § Auto-Start Removal
  - Success: Uses _resolve_server_address instead of _resolve_server_address_with_autostart
  - Test: test_cli_commands_use_resolve_server_address (new)
  - Location: ragzoom/cli.py:647

- [ ] Update call site at line 647 (other command)
  - Spec: specs/grpc-cli-architecture.md § Auto-Start Removal
  - Success: Function name updated
  - Test: test_cli_commands_use_resolve_server_address
  - Location: ragzoom/cli.py:709

- [ ] Update call site at line 709
  - Spec: specs/grpc-cli-architecture.md § Auto-Start Removal
  - Success: Function name updated
  - Test: test_cli_commands_use_resolve_server_address
  - Location: ragzoom/cli.py:987

- [ ] Update call site at line 987
  - Spec: specs/grpc-cli-architecture.md § Auto-Start Removal
  - Success: Function name updated
  - Test: test_cli_commands_use_resolve_server_address
  - Location: ragzoom/cli.py:1281

- [ ] Update call site at line 1281
  - Spec: specs/grpc-cli-architecture.md § Auto-Start Removal
  - Success: Function name updated
  - Test: test_cli_commands_use_resolve_server_address
  - Location: ragzoom/cli.py:1350

- [ ] Update call site at line 1350
  - Spec: specs/grpc-cli-architecture.md § Auto-Start Removal
  - Success: Function name updated
  - Test: test_cli_commands_use_resolve_server_address
  - Location: ragzoom/cli.py:1350

### 5.3 Remove Unused Import

- [ ] Remove ensure_server_running from daemon imports
  - Spec: specs/grpc-cli-architecture.md § Auto-Start Removal
  - Success: `ensure_server_running` no longer imported in cli.py
  - Test: grep confirms no import
  - Location: ragzoom/cli.py:40

---

## Phase 6: Pin Removal (Cleanup)

Remove obsolete pin functionality. Non-blocking, can be done in parallel with other phases.

### 6.1 Remove Pin CLI Command

- [ ] Remove `pin` CLI command
  - Spec: specs/grpc-cli-architecture.md § Pin Command Removal
  - Success: No `@cli.command()` named `pin` exists
  - Test: grep confirms no pin command
  - Location: ragzoom/cli.py

### 6.2 Remove DocumentService Pin Method

- [ ] Remove pin_node() method from DocumentService
  - Spec: specs/grpc-cli-architecture.md § Pin Command Removal
  - Success: Method no longer exists
  - Test: test_document_service_no_pin_method (new)
  - Location: ragzoom/services/document_service.py:136-151

### 6.3 Remove SystemStatus pinned_nodes Field

- [ ] Remove pinned_nodes from SystemStatus dataclass
  - Spec: specs/grpc-cli-architecture.md § Pin Command Removal
  - Success: Field removed from dataclass
  - Test: test_system_status_no_pinned_nodes (new)
  - Location: ragzoom/services/document_service.py:27-33

- [ ] Remove pinned_nodes aggregation from get_system_status()
  - Spec: specs/grpc-cli-architecture.md § Pin Command Removal
  - Success: No pinned_count() call in get_system_status
  - Test: test_system_status_no_pinned_nodes
  - Location: ragzoom/services/document_service.py:97-116

### 6.4 Remove Database Pin Column (Future Migration)

Note: Full database migration to remove is_pinned column deferred to future work.
These items track code that references the column but can remain until migration.

- [ ] Document is_pinned column as deprecated in models.py
  - Spec: specs/grpc-cli-architecture.md § Pin Command Removal
  - Success: Comment added noting deprecation
  - Test: N/A (documentation only)
  - Location: ragzoom/models.py:67

- [ ] Document is_pinned column as deprecated in sqlite_db.py
  - Spec: specs/grpc-cli-architecture.md § Pin Command Removal
  - Success: Comment added noting deprecation
  - Test: N/A (documentation only)
  - Location: ragzoom/backends/sqlite_db.py:43

---

## Phase 7: Tests

New tests to verify implementation.

### 7.1 Proto Tests

- [ ] Add test_list_documents_proto_exists
  - Spec: specs/grpc-cli-architecture.md § Testing
  - Success: Test imports ListDocumentsRequest, ListDocumentsResponse, DocumentInfo
  - Test: tests/test_grpc_proto.py::test_list_documents_proto_exists
  - Location: tests/test_grpc_proto.py (new file)

- [ ] Add test_validate_document_proto_exists
  - Spec: specs/grpc-cli-architecture.md § Testing
  - Success: Test imports ValidateDocumentRequest, ValidateDocumentResponse
  - Test: tests/test_grpc_proto.py::test_validate_document_proto_exists
  - Location: tests/test_grpc_proto.py

- [ ] Add test_get_system_status_proto_exists
  - Spec: specs/grpc-cli-architecture.md § Testing
  - Success: Test imports GetSystemStatusRequest, GetSystemStatusResponse
  - Test: tests/test_grpc_proto.py::test_get_system_status_proto_exists
  - Location: tests/test_grpc_proto.py

- [ ] Add test_get_cost_stats_proto_exists
  - Spec: specs/grpc-cli-architecture.md § Testing
  - Success: Test imports GetCostStatsRequest, GetCostStatsResponse, DocumentCostStats
  - Test: tests/test_grpc_proto.py::test_get_cost_stats_proto_exists
  - Location: tests/test_grpc_proto.py

### 7.2 Servicer Unit Tests

- [ ] Add test_list_documents_servicer
  - Spec: specs/grpc-cli-architecture.md § Unit Tests
  - Success: Servicer returns correct documents
  - Test: tests/test_grpc_servicers.py::test_list_documents_servicer
  - Location: tests/test_grpc_servicers.py (new file or extend existing)

- [ ] Add test_validate_document_servicer
  - Spec: specs/grpc-cli-architecture.md § Unit Tests
  - Success: Servicer returns valid/errors correctly
  - Test: tests/test_grpc_servicers.py::test_validate_document_servicer
  - Location: tests/test_grpc_servicers.py

- [ ] Add test_validate_document_not_found
  - Spec: specs/grpc-cli-architecture.md § Error Handling
  - Success: NOT_FOUND raised for missing document
  - Test: tests/test_grpc_servicers.py::test_validate_document_not_found
  - Location: tests/test_grpc_servicers.py

- [ ] Add test_get_system_status_servicer
  - Spec: specs/grpc-cli-architecture.md § Unit Tests
  - Success: Servicer returns aggregated stats
  - Test: tests/test_grpc_servicers.py::test_get_system_status_servicer
  - Location: tests/test_grpc_servicers.py

- [ ] Add test_get_cost_stats_servicer
  - Spec: specs/grpc-cli-architecture.md § Unit Tests
  - Success: Servicer returns cost stats for document
  - Test: tests/test_grpc_servicers.py::test_get_cost_stats_servicer
  - Location: tests/test_grpc_servicers.py

### 7.3 Client Unit Tests

- [ ] Add test_grpc_client_list_documents
  - Spec: specs/grpc-cli-architecture.md § Unit Tests
  - Success: Client method returns document list
  - Test: tests/test_grpc_client.py::test_grpc_client_list_documents
  - Location: tests/test_grpc_client.py

- [ ] Add test_grpc_client_validate_document
  - Spec: specs/grpc-cli-architecture.md § Unit Tests
  - Success: Client method returns validation result
  - Test: tests/test_grpc_client.py::test_grpc_client_validate_document
  - Location: tests/test_grpc_client.py

- [ ] Add test_grpc_client_get_system_status
  - Spec: specs/grpc-cli-architecture.md § Unit Tests
  - Success: Client method returns system status
  - Test: tests/test_grpc_client.py::test_grpc_client_get_system_status
  - Location: tests/test_grpc_client.py

- [ ] Add test_grpc_client_get_cost_stats
  - Spec: specs/grpc-cli-architecture.md § Unit Tests
  - Success: Client method returns cost stats
  - Test: tests/test_grpc_client.py::test_grpc_client_get_cost_stats
  - Location: tests/test_grpc_client.py

### 7.4 CLI Tests

- [ ] Add test_cli_documents_uses_grpc
  - Spec: specs/grpc-cli-architecture.md § Unit Tests
  - Success: Command calls gRPC client, not DocumentService directly
  - Test: tests/test_cli_grpc.py::test_cli_documents_uses_grpc
  - Location: tests/test_cli_grpc.py (new file)

- [ ] Add test_cli_server_unreachable_error
  - Spec: specs/grpc-cli-architecture.md § Unit Tests
  - Success: Clear error message shown when server not running
  - Test: tests/test_cli_grpc.py::test_cli_server_unreachable_error
  - Location: tests/test_cli_grpc.py

- [ ] Add test_no_autostart_attempted
  - Spec: specs/grpc-cli-architecture.md § Unit Tests
  - Success: ensure_server_running never called
  - Test: tests/test_cli_grpc.py::test_no_autostart_attempted
  - Location: tests/test_cli_grpc.py

### 7.5 Integration Tests

- [ ] Add test_grpc_list_documents_integration
  - Spec: specs/grpc-cli-architecture.md § Integration Tests
  - Success: Full roundtrip: index doc, list via gRPC, verify response
  - Test: tests/test_grpc_server_integration.py::test_grpc_list_documents_integration
  - Location: tests/test_grpc_server_integration.py

- [ ] Add test_grpc_validate_document_integration
  - Spec: specs/grpc-cli-architecture.md § Integration Tests
  - Success: Full roundtrip: index doc, validate via gRPC
  - Test: tests/test_grpc_server_integration.py::test_grpc_validate_document_integration
  - Location: tests/test_grpc_server_integration.py

- [ ] Add test_grpc_system_status_integration
  - Spec: specs/grpc-cli-architecture.md § Integration Tests
  - Success: Full roundtrip: index doc, get status via gRPC
  - Test: tests/test_grpc_server_integration.py::test_grpc_system_status_integration
  - Location: tests/test_grpc_server_integration.py

- [ ] Add test_grpc_cost_stats_integration
  - Spec: specs/grpc-cli-architecture.md § Integration Tests
  - Success: Full roundtrip: index doc, get cost via gRPC
  - Test: tests/test_grpc_server_integration.py::test_grpc_cost_stats_integration
  - Location: tests/test_grpc_server_integration.py

---

## Summary

| Phase | Items | Dependencies |
|-------|-------|--------------|
| 1. Proto Definitions | 15 | None |
| 2. Server Servicers | 7 | Phase 1 |
| 3. gRPC Client | 8 | Phase 2 |
| 4. CLI Migration | 10 | Phase 3 |
| 5. Auto-Start Removal | 9 | None (parallel) |
| 6. Pin Removal | 6 | None (parallel) |
| 7. Tests | 18 | Phases 1-5 |

**Total: 73 items**

Recommended execution order:
1. Phases 1-4 (core functionality, sequential)
2. Phase 5 (auto-start removal, can start after Phase 3)
3. Phase 6 (pin removal, independent, can run in parallel)
4. Phase 7 (tests, incremental with each phase)
