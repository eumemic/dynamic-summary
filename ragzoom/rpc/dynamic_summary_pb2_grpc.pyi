from __future__ import annotations

# ruff: noqa

from collections.abc import AsyncIterator, Awaitable, Iterable
from typing import Callable, NoReturn, Protocol

from .dynamic_summary_pb2 import (
    AppendTextRequest,
    AppendTextResponse,
    BatchAppendTextRequest,
    BatchAppendTextResponse,
    ExecuteQueryRequest,
    ExecuteQueryResponse,
    GetDocumentRequest,
    GetDocumentResponse,
    IndexDocumentRequest,
    IndexDocumentResponse,
    RetrieveRequest,
    RetrieveResponse,
    RunWorkersRequest,
    RunWorkersResponse,
    TruncateDocumentRequest,
    TruncateDocumentResponse,
)

Channel = object
Server = object

class ServicerContext(Protocol):
    async def abort(self, code: object, details: str) -> NoReturn: ...

class IndexerServiceStub:
    def __init__(self, channel: Channel) -> None: ...
    def IndexDocument(
        self, request: IndexDocumentRequest, timeout: float | None = ...
    ) -> IndexDocumentResponse: ...
    def AppendText(
        self, request: AppendTextRequest, timeout: float | None = ...
    ) -> AppendTextResponse: ...
    def BatchAppendText(
        self, request: BatchAppendTextRequest, timeout: float | None = ...
    ) -> BatchAppendTextResponse: ...
    def TruncateDocument(
        self, request: TruncateDocumentRequest, timeout: float | None = ...
    ) -> TruncateDocumentResponse: ...

class RetrievalServiceStub:
    def __init__(self, channel: Channel) -> None: ...
    def Retrieve(
        self, request: RetrieveRequest, timeout: float | None = ...
    ) -> RetrieveResponse: ...
    def ExecuteQuery(
        self, request: ExecuteQueryRequest, timeout: float | None = ...
    ) -> ExecuteQueryResponse: ...

class WorkerServiceStub:
    def __init__(self, channel: Channel) -> None: ...
    def RunWorkers(
        self, request: RunWorkersRequest, timeout: float | None = ...
    ) -> Iterable[RunWorkersResponse]: ...
    def GetDocument(
        self, request: GetDocumentRequest, timeout: float | None = ...
    ) -> GetDocumentResponse: ...

class IndexerServiceServicer:
    def IndexDocument(
        self, request: IndexDocumentRequest, context: ServicerContext
    ) -> Awaitable[IndexDocumentResponse]: ...
    def AppendText(
        self, request: AppendTextRequest, context: ServicerContext
    ) -> Awaitable[AppendTextResponse]: ...
    def BatchAppendText(
        self, request: BatchAppendTextRequest, context: ServicerContext
    ) -> Awaitable[BatchAppendTextResponse]: ...
    def TruncateDocument(
        self, request: TruncateDocumentRequest, context: ServicerContext
    ) -> Awaitable[TruncateDocumentResponse]: ...

class RetrievalServiceServicer:
    def Retrieve(
        self, request: RetrieveRequest, context: ServicerContext
    ) -> Awaitable[RetrieveResponse]: ...
    def ExecuteQuery(
        self, request: ExecuteQueryRequest, context: ServicerContext
    ) -> Awaitable[ExecuteQueryResponse]: ...

class WorkerServiceServicer:
    def RunWorkers(
        self, request: RunWorkersRequest, context: ServicerContext
    ) -> AsyncIterator[RunWorkersResponse]: ...
    def GetDocument(
        self, request: GetDocumentRequest, context: ServicerContext
    ) -> Awaitable[GetDocumentResponse]: ...

add_IndexerServiceServicer_to_server: Callable[[IndexerServiceServicer, Server], None]
add_RetrievalServiceServicer_to_server: Callable[
    [RetrievalServiceServicer, Server], None
]
add_WorkerServiceServicer_to_server: Callable[[WorkerServiceServicer, Server], None]
