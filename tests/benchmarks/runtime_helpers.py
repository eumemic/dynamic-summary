"""Helpers for running indexing benchmarks via IndexerRuntime."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

from openai import OpenAI

from ragzoom.config import IndexConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.vector_index import VectorIndex as _VectorIndexProtocol
from ragzoom.indexing.runtime import IndexerRuntime
from ragzoom.server.append_executor import AppendExecutor
from ragzoom.server.indexing_engine import IndexingEngine
from ragzoom.server.run_manager import TelemetryRunManager
from ragzoom.services.indexing_service import IndexingResult
from ragzoom.services.llm_service import LLMService
from ragzoom.telemetry_types import TelemetryDataDict

logger = logging.getLogger(__name__)


@dataclass
class RuntimeBundle:
    """Active runtime components for benchmark runs."""

    runtime: IndexerRuntime
    indexing_engine: IndexingEngine
    telemetry_manager: TelemetryRunManager


@contextlib.asynccontextmanager
async def create_runtime(
    *,
    storage_backend: StorageBackend,
    index_config: IndexConfig,
    vector_index: _VectorIndexProtocol,
    api_key: str,
    worker_count: int = 4,
) -> AsyncIterator[RuntimeBundle]:
    """Provision an IndexerRuntime backed by real OpenAI clients."""

    prev_pytest_flag = os.environ.pop("PYTEST_CURRENT_TEST", None)
    try:
        llm_service = LLMService(index_config, api_key=api_key)
    finally:
        if prev_pytest_flag is not None:
            os.environ["PYTEST_CURRENT_TEST"] = prev_pytest_flag

    append_executor = AppendExecutor(index_config, llm_service)
    telemetry_manager = TelemetryRunManager(index_config)

    openai_client = OpenAI(api_key=api_key)

    indexing_engine = IndexingEngine(
        store=storage_backend,
        llm_service=llm_service,
        index_config=index_config,
        openai_client=openai_client,
        vector_index_factory=lambda _model_id: vector_index,
        max_parallelism=worker_count,
    )

    runtime = IndexerRuntime(
        store=storage_backend,
        index_config=index_config,
        append_executor=append_executor,
        indexing_engine=indexing_engine,
        telemetry_manager=telemetry_manager,
        vector_index_factory=lambda _model_id: vector_index,
    )

    try:
        yield RuntimeBundle(runtime, indexing_engine, telemetry_manager)
    finally:
        try:
            await asyncio.wait_for(
                asyncio.shield(indexing_engine.wait_until_idle()),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Benchmark runtime teardown: wait_until_idle timed out; forcing shutdown"
            )
        except asyncio.CancelledError:
            logger.warning(
                "Benchmark runtime teardown interrupted by cancellation; forcing shutdown"
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception(
                "Benchmark runtime teardown: wait_until_idle failed", exc_info=exc
            )
        try:
            await asyncio.shield(indexing_engine.shutdown())
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception(
                "Benchmark runtime teardown: shutdown failed", exc_info=exc
            )
        await telemetry_manager.prune_expired()


def append_document(
    *,
    storage_backend: StorageBackend,
    index_config: IndexConfig,
    vector_index: _VectorIndexProtocol,
    document_id: str,
    text: str,
    api_key: str,
    file_path: str | None = None,
    replace_existing: bool = True,
    collect_telemetry: bool = False,
    worker_count: int = 4,
) -> tuple[IndexingResult, TelemetryDataDict | None]:
    """Append text to a document via the runtime and optionally capture telemetry."""

    async def _run() -> tuple[IndexingResult, TelemetryDataDict | None]:
        async with create_runtime(
            storage_backend=storage_backend,
            index_config=index_config,
            vector_index=vector_index,
            api_key=api_key,
            worker_count=worker_count,
        ) as bundle:
            session = bundle.runtime.get_session(document_id, file_path=file_path)
            result = await session.append_text(
                text,
                replace_existing=replace_existing,
                collect_telemetry=collect_telemetry,
            )
            await bundle.indexing_engine.wait_until_idle(document_id)

            telemetry_payload: TelemetryDataDict | None = None
            if collect_telemetry:
                run_id = result.telemetry_run_id or ""
                run_context = await bundle.telemetry_manager.get_run(run_id)
                if run_context is None:
                    run_context = await bundle.telemetry_manager.latest_for_document(
                        document_id
                    )
                if run_context is not None:
                    completed = await bundle.telemetry_manager.wait_for_completion(
                        run_context
                    )
                    telemetry_payload = completed.result
            return result, telemetry_payload

    return asyncio.run(_run())


__all__ = ["append_document"]
