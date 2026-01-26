from __future__ import annotations

# ruff: noqa

from collections.abc import AsyncIterator, Awaitable, Iterable
from typing import Callable, NoReturn, Protocol

from .dynamic_summary_pb2 import (
    AppendTextRequest,
    AppendTextResponse,
    BatchAppendTextRequest,
    BatchAppendTextResponse,
    DocumentStatusRequest,
    DocumentStatusResponse,
    ExecuteQueryRequest,
    ExecuteQueryResponse,
    GetCompactionBoundaryRequest,
    GetCompactionBoundaryResponse,
    GetDocumentRequest,
    GetDocumentResponse,
    GetSessionCursorRequest,
    GetSessionCursorResponse,
    IndexDocumentRequest,
    IndexDocumentResponse,
    IngestSessionRequest,
    IngestSessionResponse,
    ResetSessionCursorRequest,
    ResetSessionCursorResponse,
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
    def invocation_metadata(self) -> list[tuple[str, str]] | None: ...
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
    def GetDocumentStatus(
        self, request: DocumentStatusRequest, timeout: float | None = ...
    ) -> DocumentStatusResponse: ...

class SessionIngestionServiceStub:
    def __init__(self, channel: Channel) -> None: ...
    def GetSessionCursor(
        self,
        request: GetSessionCursorRequest,
        timeout: float | None = ...,
        metadata: list[tuple[str, str]] | None = ...,
    ) -> GetSessionCursorResponse: ...
    def IngestSession(
        self,
        request: IngestSessionRequest,
        timeout: float | None = ...,
        metadata: list[tuple[str, str]] | None = ...,
    ) -> IngestSessionResponse: ...
    def GetCompactionBoundary(
        self,
        request: GetCompactionBoundaryRequest,
        timeout: float | None = ...,
        metadata: list[tuple[str, str]] | None = ...,
    ) -> GetCompactionBoundaryResponse: ...
    def ResetSessionCursor(
        self,
        request: ResetSessionCursorRequest,
        timeout: float | None = ...,
        metadata: list[tuple[str, str]] | None = ...,
    ) -> ResetSessionCursorResponse: ...

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
    def GetDocumentStatus(
        self, request: object, context: ServicerContext
    ) -> Awaitable[object]: ...

class SessionIngestionServiceServicer:
    def GetSessionCursor(
        self, request: GetSessionCursorRequest, context: ServicerContext
    ) -> Awaitable[GetSessionCursorResponse]: ...
    def IngestSession(
        self, request: IngestSessionRequest, context: ServicerContext
    ) -> Awaitable[IngestSessionResponse]: ...
    def GetCompactionBoundary(
        self, request: GetCompactionBoundaryRequest, context: ServicerContext
    ) -> Awaitable[GetCompactionBoundaryResponse]: ...
    def ResetSessionCursor(
        self, request: ResetSessionCursorRequest, context: ServicerContext
    ) -> Awaitable[ResetSessionCursorResponse]: ...

add_IndexerServiceServicer_to_server: Callable[[IndexerServiceServicer, Server], None]
add_RetrievalServiceServicer_to_server: Callable[
    [RetrievalServiceServicer, Server], None
]
add_WorkerServiceServicer_to_server: Callable[[WorkerServiceServicer, Server], None]
add_SessionIngestionServiceServicer_to_server: Callable[
    [SessionIngestionServiceServicer, Server], None
]
