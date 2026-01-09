"""Integration tests for the indexer lease mechanism.

These tests verify lease behavior in more realistic scenarios:
1. Two servers - only one wins the lease
2. Failover on crash - second server takes over after first dies
3. Server startup with lease already held
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from ragzoom.server.lease import IndexerLease, LeaseConfig


@pytest.fixture
def integration_engine() -> Engine:
    """Create an in-memory SQLite engine with lease table for integration tests."""
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
        ttl_seconds=0.5,
        heartbeat_interval=0.1,
        acquire_timeout=1.0,
        acquire_poll_interval=0.1,
    )


@pytest.mark.integration
class TestLeaseIntegration:
    """Integration tests for lease coordination between multiple instances."""

    @pytest.mark.asyncio
    async def test_two_servers_one_wins(
        self, integration_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Start two servers concurrently; one acquires, one blocks."""
        server1 = IndexerLease(integration_engine, fast_config)
        server2 = IndexerLease(integration_engine, fast_config)

        # Race to acquire
        results = await asyncio.gather(
            server1.acquire(),
            server2.acquire(),
        )

        # Exactly one should succeed
        assert sum(results) == 1, f"Expected exactly 1 winner, got {results}"

        # One is acquired, one is not
        assert server1.is_acquired != server2.is_acquired

        # Clean up
        if server1.is_acquired:
            await server1.release()
        if server2.is_acquired:
            await server2.release()

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(5.0)  # Allow longer for this test
    async def test_failover_on_crash(
        self, integration_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Second server takes over after first dies without releasing."""
        server1 = IndexerLease(integration_engine, fast_config)
        server2 = IndexerLease(integration_engine, fast_config)

        # Server 1 acquires
        result1 = await server1.acquire()
        assert result1 is True

        # Simulate crash - cancel heartbeat without releasing
        if server1._heartbeat_task:
            server1._heartbeat_task.cancel()
            try:
                await server1._heartbeat_task
            except asyncio.CancelledError:
                pass
            server1._heartbeat_task = None

        # Server 2 initially can't acquire (lease not expired yet)
        # Use very short timeout to just check current state
        short_config = LeaseConfig(
            ttl_seconds=fast_config.ttl_seconds,
            heartbeat_interval=fast_config.heartbeat_interval,
            acquire_timeout=0.2,  # Very short
            acquire_poll_interval=0.05,
        )
        server2_impatient = IndexerLease(integration_engine, short_config)
        result2_early = await server2_impatient.acquire()
        assert result2_early is False, "Should not acquire while lease is active"

        # Wait for TTL to expire
        await asyncio.sleep(fast_config.ttl_seconds + 0.2)

        # Now server 2 should be able to take over
        result2 = await server2.acquire()
        assert result2 is True
        assert server2.is_acquired

        # Verify server2 is the holder
        with integration_engine.connect() as conn:
            row = conn.execute(
                text("SELECT holder_id FROM indexer_leases WHERE id = 1")
            ).fetchone()
            assert row is not None
            assert row[0] == server2.holder_id

        await server2.release()

    @pytest.mark.asyncio
    async def test_graceful_release_allows_immediate_acquisition(
        self, integration_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """When server releases gracefully, new server acquires immediately."""
        server1 = IndexerLease(integration_engine, fast_config)
        server2 = IndexerLease(integration_engine, fast_config)

        # Server 1 acquires and releases
        await server1.acquire()
        await server1.release()

        # Server 2 should acquire immediately (no waiting for TTL)
        start_time = asyncio.get_event_loop().time()
        result = await server2.acquire()
        elapsed = asyncio.get_event_loop().time() - start_time

        assert result is True
        assert elapsed < 0.2, f"Acquisition took too long: {elapsed:.2f}s"

        await server2.release()

    @pytest.mark.asyncio
    async def test_server_with_stale_lease_in_db(
        self, integration_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Server can acquire when there's an already-expired lease in DB."""
        # Pre-populate with an expired lease from a "crashed" server
        now = datetime.utcnow()
        expired_time = now - timedelta(seconds=10)  # Long expired

        with integration_engine.begin() as conn:
            conn.execute(
                text(
                    """
                INSERT INTO indexer_leases (id, holder_id, acquired_at, last_heartbeat, expires_at)
                VALUES (1, 'dead-server-id', :now, :now, :expires)
            """
                ),
                {"now": expired_time, "expires": expired_time},
            )

        # New server should steal the expired lease immediately
        server = IndexerLease(integration_engine, fast_config)
        result = await server.acquire()

        assert result is True
        assert server.is_acquired

        # Verify it's the new holder
        with integration_engine.connect() as conn:
            row = conn.execute(
                text("SELECT holder_id FROM indexer_leases WHERE id = 1")
            ).fetchone()
            assert row is not None
            assert row[0] == server.holder_id
            assert row[0] != "dead-server-id"

        await server.release()

    @pytest.mark.asyncio
    async def test_heartbeat_keeps_lease_alive(
        self, integration_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Lease stays alive when heartbeat is running, even past original TTL."""
        server = IndexerLease(integration_engine, fast_config)

        await server.acquire()

        # Wait longer than TTL
        await asyncio.sleep(fast_config.ttl_seconds * 2)

        # Lease should still be acquired (heartbeat kept it alive)
        assert server.is_acquired

        # Verify in database that expires_at was updated
        with integration_engine.connect() as conn:
            row = conn.execute(
                text("SELECT expires_at FROM indexer_leases WHERE id = 1")
            ).fetchone()
            assert row is not None
            # expires_at should be in the future
            expires_str = row[0]
            if isinstance(expires_str, str):
                expires_at = datetime.fromisoformat(expires_str)
            else:
                expires_at = expires_str
            assert expires_at > datetime.utcnow()

        await server.release()

    @pytest.mark.asyncio
    async def test_multiple_acquisition_attempts(
        self, integration_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Multiple servers trying to acquire in sequence."""
        servers = [IndexerLease(integration_engine, fast_config) for _ in range(3)]

        # First server wins
        assert await servers[0].acquire() is True

        # Others fail
        short_config = LeaseConfig(
            ttl_seconds=fast_config.ttl_seconds,
            heartbeat_interval=fast_config.heartbeat_interval,
            acquire_timeout=0.3,
            acquire_poll_interval=0.05,
        )

        for i, server in enumerate(servers[1:], 1):
            server_impatient = IndexerLease(integration_engine, short_config)
            result = await server_impatient.acquire()
            assert result is False, f"Server {i} should not acquire"

        # First server releases
        await servers[0].release()

        # Now second server can acquire
        assert await servers[1].acquire() is True

        await servers[1].release()
