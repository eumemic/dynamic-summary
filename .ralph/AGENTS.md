# Operational Knowledge

## Proto Compilation

Regenerate protobuf stubs after modifying `proto/dynamic_summary.proto`:
```bash
./scripts/compile-proto.sh
```

This generates:
- `ragzoom/rpc/dynamic_summary_pb2.py` - message classes
- `ragzoom/rpc/dynamic_summary_pb2_grpc.py` - service stubs

The `.pyi` type stub files are manually maintained for better type hints.

## Testing

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/test_grpc_proto.py
```

## Development Server

Start dev server (port 50052):
```bash
python -m ragzoom.cli server start
```

Production uses port 50051 via `ragzoom server start`.
