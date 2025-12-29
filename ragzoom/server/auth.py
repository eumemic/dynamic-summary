"""Authentication utilities for the gRPC server.

This module provides API key validation and user context extraction
for multi-tenant operation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import grpc
from sqlalchemy import select
from sqlalchemy.orm import Session

from ragzoom.models import User

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

# Metadata key for API key (gRPC uses lowercase header names)
API_KEY_HEADER = "x-api-key"


@dataclass
class AuthContext:
    """Authentication context extracted from request."""

    user_id: str
    api_key: str


class AuthError(Exception):
    """Authentication error."""

    def __init__(
        self, message: str, code: grpc.StatusCode = grpc.StatusCode.UNAUTHENTICATED
    ):
        self.message = message
        self.code = code
        super().__init__(message)


def extract_api_key(context: grpc.ServicerContext) -> str | None:
    """Extract API key from gRPC request metadata.

    Args:
        context: gRPC servicer context

    Returns:
        API key string if present, None otherwise
    """
    metadata = dict(context.invocation_metadata())
    value = metadata.get(API_KEY_HEADER)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def validate_api_key(
    api_key: str,
    session_factory: sessionmaker[Session],
) -> AuthContext:
    """Validate API key and return auth context.

    Args:
        api_key: The API key to validate
        session_factory: SQLAlchemy session factory

    Returns:
        AuthContext with user information

    Raises:
        AuthError: If API key is invalid
    """
    with session_factory() as session:
        user = session.execute(
            select(User).where(User.api_key == api_key)
        ).scalar_one_or_none()

        if user is None:
            raise AuthError("Invalid API key")

        return AuthContext(user_id=user.id, api_key=api_key)


def require_auth(
    context: grpc.ServicerContext,
    session_factory: sessionmaker[Session],
    *,
    allow_anonymous: bool = False,
) -> AuthContext | None:
    """Extract and validate authentication from request.

    Args:
        context: gRPC servicer context
        session_factory: SQLAlchemy session factory
        allow_anonymous: If True, return None for unauthenticated requests
                        instead of raising an error

    Returns:
        AuthContext if authenticated, None if anonymous allowed

    Raises:
        AuthError: If authentication is required but missing/invalid
    """
    api_key = extract_api_key(context)

    if api_key is None:
        if allow_anonymous:
            return None
        raise AuthError("API key required")

    return validate_api_key(api_key, session_factory)
