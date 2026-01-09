"""Global indexer lease mechanism for single-writer coordination.

This module provides a lease-based coordination mechanism that ensures only one
IndexingEngine instance can write to the database at a time. This prevents
corruption during deployments where old and new containers briefly run
simultaneously (e.g., Railway blue/green deployments).

The lease uses a database table with TTL-based expiration and heartbeat refresh:
- Startup: Server generates unique holder_id, blocks trying to acquire lease
- Acquisition: INSERT if empty, or UPDATE if expired (PostgreSQL uses SELECT FOR UPDATE)
- Heartbeat: Background task refreshes expires_at periodically
- Shutdown: DELETE the lease row for immediate handoff
- Crash: Lease expires after TTL, next server can claim it

Works with both PostgreSQL and SQLite backends.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection, Engine

logger = logging.getLogger(__name__)


@dataclass
class LeaseConfig:
    """Configuration for indexer lease."""

    ttl_seconds: float = 60.0
    """Seconds before lease expires without heartbeat."""

    heartbeat_interval: float = 15.0
    """Seconds between heartbeat updates."""

    acquire_timeout: float = 90.0
    """Max seconds to wait for lease acquisition."""

    acquire_poll_interval: float = 2.0
    """Seconds between acquisition attempts."""


class IndexerLease:
    """Global lease ensuring single-indexer ownership.

    Only one IndexerLease holder can be active at a time for a given database.
    The lease uses a singleton row pattern with TTL-based expiration.

    Usage:
        lease = IndexerLease(engine)
        if await lease.acquire():
            try:
                # Run server...
            finally:
                await lease.release()
        else:
            sys.exit(1)  # Let Railway restart us
    """

    def __init__(
        self,
        engine: Engine,
        config: LeaseConfig | None = None,
    ) -> None:
        """Initialize the lease manager.

        Args:
            engine: SQLAlchemy engine for database access.
            config: Optional lease configuration. Uses defaults if not provided.
        """
        self._engine = engine
        self._config = config or LeaseConfig()
        self._holder_id = str(uuid.uuid4())
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._acquired = False
        # Detect SQLite for dialect-specific SQL
        self._is_sqlite = engine.dialect.name == "sqlite"

    @property
    def holder_id(self) -> str:
        """Unique identifier for this lease holder."""
        return self._holder_id

    @property
    def is_acquired(self) -> bool:
        """Whether this instance currently holds the lease."""
        return self._acquired

    async def acquire(self) -> bool:
        """Block until lease acquired or timeout.

        Returns:
            True if lease was acquired, False if timed out.
        """
        deadline = time.monotonic() + self._config.acquire_timeout
        attempt = 0

        while time.monotonic() < deadline:
            attempt += 1
            try:
                with self._engine.begin() as conn:
                    if self._try_acquire_or_steal_expired(conn):
                        self._acquired = True
                        self._heartbeat_task = asyncio.create_task(
                            self._heartbeat_loop(),
                            name=f"lease-heartbeat:{self._holder_id[:8]}",
                        )
                        logger.info(
                            "Acquired indexer lease (holder=%s)",
                            self._holder_id[:8],
                        )
                        return True
            except Exception:
                logger.exception("Error during lease acquisition attempt %d", attempt)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            wait_time = min(self._config.acquire_poll_interval, remaining)
            logger.debug(
                "Lease held by another instance, retrying in %.1fs (attempt %d)",
                wait_time,
                attempt,
            )
            await asyncio.sleep(wait_time)

        logger.error(
            "Failed to acquire indexer lease after %.0fs (%d attempts)",
            self._config.acquire_timeout,
            attempt,
        )
        return False

    async def release(self) -> None:
        """Release the lease gracefully."""
        # Stop heartbeat first
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None

        if not self._acquired:
            return

        try:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text(
                        """
                        DELETE FROM indexer_leases
                        WHERE id = 1 AND holder_id = :holder_id
                    """
                    ),
                    {"holder_id": self._holder_id},
                )
                if result.rowcount > 0:
                    logger.info(
                        "Released indexer lease (holder=%s)", self._holder_id[:8]
                    )
                else:
                    logger.warning(
                        "Lease was not held by us during release (holder=%s)",
                        self._holder_id[:8],
                    )
        except Exception:
            logger.exception("Failed to release lease cleanly")
        finally:
            self._acquired = False

    def _try_acquire_or_steal_expired(self, conn: Connection) -> bool:
        """Attempt to acquire the lease atomically.

        Uses SELECT FOR UPDATE on PostgreSQL to prevent races between instances.
        SQLite uses transaction-level isolation (no FOR UPDATE needed).

        Args:
            conn: Database connection (should be in a transaction).

        Returns:
            True if lease was acquired, False if held by another active instance.
        """
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=self._config.ttl_seconds)

        # Lock the row (or nothing if no row exists)
        # SQLite doesn't support FOR UPDATE but transactions provide isolation
        if self._is_sqlite:
            select_sql = """
                SELECT holder_id, expires_at
                FROM indexer_leases
                WHERE id = 1
            """
        else:
            select_sql = """
                SELECT holder_id, expires_at
                FROM indexer_leases
                WHERE id = 1
                FOR UPDATE
            """
        result = conn.execute(text(select_sql))
        row = result.fetchone()

        if row is None:
            # No lease exists - create one
            conn.execute(
                text(
                    """
                    INSERT INTO indexer_leases
                    (id, holder_id, acquired_at, last_heartbeat, expires_at)
                    VALUES (1, :holder_id, :now, :now, :expires_at)
                """
                ),
                {
                    "holder_id": self._holder_id,
                    "now": now,
                    "expires_at": expires_at,
                },
            )
            return True

        current_holder, current_expires = row

        if current_holder == self._holder_id:
            # We already hold it (shouldn't happen normally, but be safe)
            logger.debug("Lease already held by us")
            return True

        # SQLite stores timestamps as strings, PostgreSQL as datetime
        if isinstance(current_expires, str):
            current_expires = datetime.fromisoformat(current_expires)
        # Ensure timezone-aware comparison (stored times are UTC)
        if current_expires.tzinfo is None:
            current_expires = current_expires.replace(tzinfo=timezone.utc)

        if current_expires < now:
            # Lease expired - steal it
            conn.execute(
                text(
                    """
                    UPDATE indexer_leases
                    SET holder_id = :holder_id,
                        acquired_at = :now,
                        last_heartbeat = :now,
                        expires_at = :expires_at
                    WHERE id = 1
                """
                ),
                {
                    "holder_id": self._holder_id,
                    "now": now,
                    "expires_at": expires_at,
                },
            )
            logger.info(
                "Took over expired lease from %s (expired %s)",
                current_holder[:8] if current_holder else "unknown",
                current_expires,
            )
            return True

        # Lease held by another active instance
        logger.debug(
            "Lease held by %s until %s",
            current_holder[:8] if current_holder else "unknown",
            current_expires,
        )
        return False

    async def _heartbeat_loop(self) -> None:
        """Periodically refresh the lease TTL."""
        while True:
            try:
                await asyncio.sleep(self._config.heartbeat_interval)

                with self._engine.begin() as conn:
                    now = datetime.now(timezone.utc)
                    expires_at = now + timedelta(seconds=self._config.ttl_seconds)

                    # SQLite doesn't support RETURNING, use rowcount instead
                    result = conn.execute(
                        text(
                            """
                            UPDATE indexer_leases
                            SET last_heartbeat = :now, expires_at = :expires_at
                            WHERE id = 1 AND holder_id = :holder_id
                        """
                        ),
                        {
                            "now": now,
                            "expires_at": expires_at,
                            "holder_id": self._holder_id,
                        },
                    )

                    if result.rowcount == 0:
                        # Lost the lease! Someone stole it (shouldn't happen
                        # unless heartbeat interval > TTL or DB issues)
                        logger.critical(
                            "Lost indexer lease! Another instance may be active. "
                            "(holder=%s)",
                            self._holder_id[:8],
                        )
                        self._acquired = False
                        return

                    logger.debug(
                        "Heartbeat sent (holder=%s, expires=%s)",
                        self._holder_id[:8],
                        expires_at,
                    )

            except asyncio.CancelledError:
                raise
            except Exception:
                # Log but continue - transient DB issues shouldn't kill heartbeat
                logger.exception("Heartbeat failed, will retry")
