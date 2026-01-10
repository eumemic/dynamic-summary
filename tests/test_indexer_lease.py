"""Unit tests for the indexer lease mechanism.

Tests validate that the IndexerLease class correctly:
1. Acquires a fresh lease when table is empty
2. Blocks when lease is held by another instance
3. Steals expired leases after TTL
4. Maintains lease via heartbeat
5. Releases lease gracefully on shutdown
6. Returns False on acquisition timeout
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from ragzoom.server.lease import IndexerLease, LeaseConfig


@pytest.fixture
def lease_engine() -> Engine:
    """Create an in-memory SQLite engine with lease table for testing."""
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
            CREATE TABLE indexer_leases (
                id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
                holder_id VARCHAR(255) NOT NULL,
                acquired_at TIMESTAMP NOT NULL,
                last_heartbeat TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL
            )
        """
            )
        )
    return engine


@pytest.fixture
def fast_config() -> LeaseConfig:
    """Config with short timeouts for fast tests."""
    return LeaseConfig(
        ttl_seconds=1.0,
        heartbeat_interval=0.2,
        acquire_timeout=0.5,
        acquire_poll_interval=0.1,
    )


class TestLeaseConfig:
    """Test LeaseConfig defaults and validation."""

    def test_default_values(self) -> None:
        """Default config has reasonable production values."""
        config = LeaseConfig()
        assert config.ttl_seconds == 60.0
        assert config.heartbeat_interval == 15.0
        assert config.acquire_timeout == 90.0
        assert config.acquire_poll_interval == 2.0

    def test_custom_values(self) -> None:
        """Custom values are preserved."""
        config = LeaseConfig(
            ttl_seconds=30.0,
            heartbeat_interval=5.0,
            acquire_timeout=45.0,
            acquire_poll_interval=1.0,
        )
        assert config.ttl_seconds == 30.0
        assert config.heartbeat_interval == 5.0
        assert config.acquire_timeout == 45.0
        assert config.acquire_poll_interval == 1.0


class TestIndexerLease:
    """Test IndexerLease acquisition and release."""

    def test_holder_id_is_unique(self, lease_engine: Engine) -> None:
        """Each lease instance gets a unique holder ID."""
        lease1 = IndexerLease(lease_engine)
        lease2 = IndexerLease(lease_engine)
        assert lease1.holder_id != lease2.holder_id
        assert len(lease1.holder_id) == 36  # UUID format

    def test_initial_state(self, lease_engine: Engine) -> None:
        """Lease starts in non-acquired state."""
        lease = IndexerLease(lease_engine)
        assert not lease.is_acquired

    @pytest.mark.asyncio
    async def test_acquire_fresh_lease(
        self, lease_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Acquire lease when table is empty."""
        lease = IndexerLease(lease_engine, fast_config)

        acquired = await lease.acquire()

        assert acquired is True
        assert lease.is_acquired

        # Verify database state
        with lease_engine.connect() as conn:
            result = conn.execute(
                text("SELECT holder_id FROM indexer_leases WHERE id = 1")
            )
            row = result.fetchone()
            assert row is not None
            assert row[0] == lease.holder_id

        await lease.release()

    @pytest.mark.asyncio
    async def test_release_clears_lease(
        self, lease_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Release deletes the lease row."""
        lease = IndexerLease(lease_engine, fast_config)
        await lease.acquire()

        await lease.release()

        assert not lease.is_acquired

        # Verify row is deleted
        with lease_engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM indexer_leases"))
            count = result.scalar()
            assert count == 0

    @pytest.mark.asyncio
    async def test_release_without_acquire_is_safe(
        self, lease_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Release on non-acquired lease is a no-op."""
        lease = IndexerLease(lease_engine, fast_config)

        # Should not raise
        await lease.release()
        assert not lease.is_acquired

    @pytest.mark.asyncio
    async def test_acquire_blocks_when_held(
        self, lease_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Second instance blocks while first holds lease."""
        lease1 = IndexerLease(lease_engine, fast_config)
        lease2 = IndexerLease(lease_engine, fast_config)

        # First instance acquires
        await lease1.acquire()

        # Second instance times out
        acquired = await lease2.acquire()
        assert acquired is False
        assert not lease2.is_acquired

        # First still holds
        assert lease1.is_acquired

        await lease1.release()

    @pytest.mark.asyncio
    async def test_acquire_steals_expired_lease(
        self, lease_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Lease can be stolen when previous holder's TTL expires."""
        # Insert an already-expired lease
        expired_time = datetime.now(timezone.utc) - timedelta(seconds=10)
        with lease_engine.begin() as conn:
            conn.execute(
                text(
                    """
                INSERT INTO indexer_leases
                (id, holder_id, acquired_at, last_heartbeat, expires_at)
                VALUES (1, 'old-holder', :now, :now, :expired)
            """
                ),
                {"now": expired_time, "expired": expired_time},
            )

        # New lease should steal it
        lease = IndexerLease(lease_engine, fast_config)
        acquired = await lease.acquire()

        assert acquired is True
        assert lease.is_acquired

        # Verify we're now the holder
        with lease_engine.connect() as conn:
            result = conn.execute(
                text("SELECT holder_id FROM indexer_leases WHERE id = 1")
            )
            row = result.fetchone()
            assert row is not None
            assert row[0] == lease.holder_id

        await lease.release()

    @pytest.mark.asyncio
    async def test_steal_expired_lease_live(
        self, lease_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Take over lease after TTL expires (live wait)."""
        lease1 = IndexerLease(lease_engine, fast_config)
        lease2 = IndexerLease(lease_engine, fast_config)

        # First instance acquires
        await lease1.acquire()
        # Stop heartbeat so it expires
        if lease1._heartbeat_task:
            lease1._heartbeat_task.cancel()
            try:
                await lease1._heartbeat_task
            except asyncio.CancelledError:
                pass
            lease1._heartbeat_task = None

        # Wait for TTL to expire
        await asyncio.sleep(fast_config.ttl_seconds + 0.2)

        # Second instance should steal it
        acquired = await lease2.acquire()
        assert acquired is True
        assert lease2.is_acquired

        # Verify new holder
        with lease_engine.connect() as conn:
            result = conn.execute(
                text("SELECT holder_id FROM indexer_leases WHERE id = 1")
            )
            row = result.fetchone()
            assert row is not None
            assert row[0] == lease2.holder_id

        await lease2.release()

    @pytest.mark.asyncio
    async def test_heartbeat_extends_ttl(
        self, lease_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Heartbeat updates expires_at to prevent expiration."""
        lease = IndexerLease(lease_engine, fast_config)
        await lease.acquire()

        # Get initial expiration
        with lease_engine.connect() as conn:
            result = conn.execute(
                text("SELECT expires_at FROM indexer_leases WHERE id = 1")
            )
            initial_expires = result.scalar()

        # Wait for heartbeat
        await asyncio.sleep(fast_config.heartbeat_interval + 0.1)

        # Check expiration was extended
        with lease_engine.connect() as conn:
            result = conn.execute(
                text("SELECT expires_at FROM indexer_leases WHERE id = 1")
            )
            new_expires = result.scalar()

        # Parse timestamps if needed (SQLite returns strings)
        if isinstance(initial_expires, str):
            initial_expires = datetime.fromisoformat(initial_expires)
        if isinstance(new_expires, str):
            new_expires = datetime.fromisoformat(new_expires)

        assert initial_expires is not None
        assert new_expires is not None
        assert new_expires > initial_expires

        await lease.release()

    @pytest.mark.asyncio
    async def test_timeout_returns_false(
        self, lease_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Returns False on timeout, doesn't raise."""
        lease1 = IndexerLease(lease_engine, fast_config)
        lease2 = IndexerLease(lease_engine, fast_config)

        await lease1.acquire()

        # Second should timeout and return False
        result = await lease2.acquire()
        assert result is False
        assert not lease2.is_acquired

        await lease1.release()

    @pytest.mark.asyncio
    async def test_reacquire_own_lease(
        self, lease_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Re-acquiring our own lease succeeds immediately."""
        lease = IndexerLease(lease_engine, fast_config)

        await lease.acquire()

        # Stop heartbeat
        if lease._heartbeat_task:
            lease._heartbeat_task.cancel()
            try:
                await lease._heartbeat_task
            except asyncio.CancelledError:
                pass
            lease._heartbeat_task = None

        # Re-acquire with same lease object
        # This tests the "already hold it" branch
        with lease_engine.begin() as conn:
            result = lease._try_acquire_or_steal_expired(conn)
            assert result is True

        await lease.release()

    @pytest.mark.asyncio
    async def test_release_stops_heartbeat(
        self, lease_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Release cancels the heartbeat task."""
        lease = IndexerLease(lease_engine, fast_config)
        await lease.acquire()

        assert lease._heartbeat_task is not None
        heartbeat_task = lease._heartbeat_task

        await lease.release()

        assert lease._heartbeat_task is None
        assert heartbeat_task.cancelled() or heartbeat_task.done()

    @pytest.mark.asyncio
    async def test_acquire_handles_db_error(
        self, lease_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Acquisition retries on transient DB errors and eventually succeeds."""
        lease = IndexerLease(lease_engine, fast_config)

        call_count = 0
        original_begin = lease_engine.begin

        def failing_begin(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Transient DB error")
            return original_begin(*args, **kwargs)

        with patch.object(lease_engine, "begin", side_effect=failing_begin):
            # Should retry and eventually succeed on 3rd attempt
            result = await lease.acquire()

        # Retried through errors and eventually succeeded
        assert result is True
        assert call_count == 3  # Failed 2 times, succeeded on 3rd

        await lease.release()

    @pytest.mark.asyncio
    async def test_second_lease_acquires_after_release(
        self, lease_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Second lease can acquire after first releases."""
        lease1 = IndexerLease(lease_engine, fast_config)
        lease2 = IndexerLease(lease_engine, fast_config)

        await lease1.acquire()
        await lease1.release()

        acquired = await lease2.acquire()
        assert acquired is True
        assert lease2.is_acquired

        await lease2.release()


class TestHeartbeatLoop:
    """Test heartbeat loop behavior."""

    @pytest.mark.asyncio
    async def test_heartbeat_continues_on_transient_error(
        self, lease_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Heartbeat logs but continues on transient errors."""
        lease = IndexerLease(lease_engine, fast_config)
        await lease.acquire()

        error_count = 0
        original_begin = lease_engine.begin

        def sometimes_failing_begin(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal error_count
            error_count += 1
            if error_count == 2:
                raise Exception("Transient error")
            return original_begin(*args, **kwargs)

        with patch.object(lease_engine, "begin", side_effect=sometimes_failing_begin):
            # Wait for multiple heartbeats
            await asyncio.sleep(fast_config.heartbeat_interval * 3)

        # Lease should still be acquired (heartbeat recovered)
        assert lease.is_acquired

        await lease.release()

    @pytest.mark.asyncio
    async def test_heartbeat_detects_lost_lease(
        self, lease_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Heartbeat detects when lease was stolen."""
        lease = IndexerLease(lease_engine, fast_config)
        await lease.acquire()

        # Simulate another instance stealing the lease
        with lease_engine.begin() as conn:
            conn.execute(
                text(
                    """
                UPDATE indexer_leases
                SET holder_id = 'other-holder'
                WHERE id = 1
            """
                )
            )

        # Wait for heartbeat to detect the theft
        await asyncio.sleep(fast_config.heartbeat_interval + 0.2)

        # Lease should mark itself as not acquired
        # (heartbeat UPDATE returns no rows)
        assert not lease.is_acquired

        # Cleanup - heartbeat task should have stopped
        if lease._heartbeat_task:
            lease._heartbeat_task.cancel()
            try:
                await lease._heartbeat_task
            except asyncio.CancelledError:
                pass


class TestLeaseTableSchema:
    """Test lease table constraints and schema."""

    def test_singleton_constraint(self, lease_engine: Engine) -> None:
        """Only one row can exist (id must be 1)."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=60)

        with lease_engine.begin() as conn:
            # First insert succeeds
            conn.execute(
                text(
                    """
                INSERT INTO indexer_leases (id, holder_id, acquired_at, last_heartbeat, expires_at)
                VALUES (1, 'holder-1', :now, :now, :expires)
            """
                ),
                {"now": now, "expires": expires},
            )

        # Second insert with id=1 fails
        with pytest.raises(Exception):  # IntegrityError
            with lease_engine.begin() as conn:
                conn.execute(
                    text(
                        """
                    INSERT INTO indexer_leases (id, holder_id, acquired_at, last_heartbeat, expires_at)
                    VALUES (1, 'holder-2', :now, :now, :expires)
                """
                    ),
                    {"now": now, "expires": expires},
                )

    def test_check_constraint_id_equals_1(self, lease_engine: Engine) -> None:
        """Cannot insert row with id != 1."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=60)

        with pytest.raises(Exception):  # Check constraint violation
            with lease_engine.begin() as conn:
                conn.execute(
                    text(
                        """
                    INSERT INTO indexer_leases (id, holder_id, acquired_at, last_heartbeat, expires_at)
                    VALUES (2, 'holder-1', :now, :now, :expires)
                """
                    ),
                    {"now": now, "expires": expires},
                )
