"""Server runtime for RagZoom gRPC deployment."""

from __future__ import annotations

from importlib import import_module

__all__ = ["ServerOptions", "ServerState", "run_server"]


def __getattr__(name: str) -> object:
    if name in {"ServerOptions", "run_server"}:
        module = import_module("ragzoom.server.app")
        return getattr(module, name)
    if name == "ServerState":
        module = import_module("ragzoom.server.state")
        return getattr(module, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(__all__)
