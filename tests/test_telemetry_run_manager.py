import asyncio
import json
from types import SimpleNamespace
from typing import NoReturn, cast

import grpc
import pytest

from ragzoom.config import IndexConfig
from ragzoom.rpc import dynamic_summary_pb2 as pb2
from ragzoom.server.run_manager import TelemetryRunManager
from ragzoom.server.servicers import WorkerServicer
from ragzoom.server.state import ServerState


class StubContext:
    def __init__(self) -> None:
        self.code: grpc.StatusCode | None = None
        self.details: str | None = None

    async def abort(self, code: object, details: str) -> NoReturn:
        self.code = cast(grpc.StatusCode, code)
        self.details = details
        raise RuntimeError("aborted")


@pytest.mark.asyncio
async def test_telemetry_run_manager_completes_and_finalizes() -> None:
    index_config = IndexConfig.load()
    manager = TelemetryRunManager(index_config)

    context = await manager.start_run(
        "doc",
        collect=True,
        source_tokens=120,
        document_path=None,
    )

    context.register_append_outcome(
        span_start=0,
        span_end=120,
        mutated_nodes=5,
        new_leaves=3,
        previous_leaf_count=2,
        total_leaves=5,
    )
    context.register_summary_node("parent-1")

    waiter = asyncio.create_task(manager.wait_for_completion(context))
    await manager.complete_run(context.run_id)
    awaited = await waiter

    assert awaited.status == "completed"
    assert awaited.error is None
    assert awaited.result is not None

    append_metadata = awaited.result.get("append_metadata")
    assert append_metadata is not None
    assert append_metadata["summary_nodes"] == 1
    assert append_metadata["leaf_delta"] == 3


@pytest.mark.asyncio
async def test_telemetry_run_manager_marks_failure() -> None:
    index_config = IndexConfig.load()
    manager = TelemetryRunManager(index_config)
    context = await manager.start_run(
        "doc",
        collect=True,
        source_tokens=10,
        document_path=None,
    )

    await manager.complete_run(context.run_id, error="boom")
    stored = await manager.get_run(context.run_id)
    assert stored is not None
    assert stored.status == "failed"
    assert stored.error == "boom"


@pytest.mark.asyncio
async def test_get_telemetry_waits_until_complete() -> None:
    index_config = IndexConfig.load()
    manager = TelemetryRunManager(index_config)
    state = cast(ServerState, SimpleNamespace(telemetry_run_manager=manager))
    servicer = WorkerServicer(state)
    context_stub = StubContext()

    run = await manager.start_run(
        "doc",
        collect=True,
        source_tokens=42,
        document_path=None,
    )
    run.register_append_outcome(
        span_start=0,
        span_end=42,
        mutated_nodes=4,
        new_leaves=3,
        previous_leaf_count=1,
        total_leaves=4,
    )
    run.register_summary_node("parent")

    request_cls = getattr(pb2, "GetTelemetryRequest")
    incomplete = await servicer.GetTelemetry(
        request_cls(document_id="doc", run_id=run.run_id, wait=False),
        context_stub,
    )
    assert not incomplete.complete
    assert incomplete.telemetry_json == ""
    assert incomplete.error == ""

    async def finalize() -> None:
        await asyncio.sleep(0.01)
        await manager.complete_run(run.run_id)

    asyncio.create_task(finalize())
    complete = await servicer.GetTelemetry(
        request_cls(document_id="doc", run_id=run.run_id, wait=True),
        StubContext(),
    )
    assert complete.complete
    assert complete.error == ""
    telemetry_payload = json.loads(complete.telemetry_json)
    assert telemetry_payload["append_metadata"]["summary_nodes"] == 1


@pytest.mark.asyncio
async def test_get_telemetry_not_found_aborts() -> None:
    index_config = IndexConfig.load()
    manager = TelemetryRunManager(index_config)
    state = cast(ServerState, SimpleNamespace(telemetry_run_manager=manager))
    servicer = WorkerServicer(state)
    context = StubContext()
    request_cls = getattr(pb2, "GetTelemetryRequest")

    with pytest.raises(RuntimeError):
        await servicer.GetTelemetry(
            request_cls(document_id="doc", run_id="missing", wait=False),
            context,
        )

    assert context.code == grpc.StatusCode.NOT_FOUND
    assert context.details == "Telemetry run not found."
