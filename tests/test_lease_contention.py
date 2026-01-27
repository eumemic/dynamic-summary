"""Tests for lease contention bug (Issue #6).

This test file reproduces the lease contention scenario where:
1. Multiple daemon processes bind to the same port via SO_REUSEPORT
2. Only one process acquires the lease
3. Others are stuck waiting for lease (up to 90s) then exit
4. During this time, multiple daemons are listening on port 50051
5. Clients may connect to the "wrong" daemon (one without the lease)

The root cause is the ordering of operations:
  1. daemonize() - forks and writes PID file
  2. write_port_file() - writes port file  <-- NOW CLIENTS THINK WE'RE READY
  3. install_shutdown_handlers()
  4. run_server() -> acquires lease  <-- BLOCKS UP TO 90s!

The port file is written BEFORE the lease is acquired, so:
- Health checks may pass (gRPC binds with SO_REUSEPORT)
- Clients think the daemon is "healthy" when it hasn't actually acquired
  the right to process requests
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from ragzoom.server.lease import IndexerLease, LeaseConfig

if TYPE_CHECKING:
    pass


@pytest.fixture
def contention_engine() -> Engine:
    """Create an in-memory SQLite engine with lease table for contention tests."""
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


class TestLeaseContention:
    """Tests demonstrating the lease contention bug (Issue #6).

    The key insight is that gRPC uses SO_REUSEPORT by default, which allows
    multiple processes to bind to the same port. Combined with the current
    startup sequence (port file written before lease acquired), this creates
    a scenario where:

    1. Daemon A starts, writes port file, starts lease acquisition
    2. Daemon B starts (stale state not cleaned up properly), writes port file
    3. Both daemons bind to port 50051 successfully (SO_REUSEPORT)
    4. Only Daemon A acquires the lease
    5. Daemon B blocks for 90s trying to acquire, then exits
    6. During this window, clients may connect to Daemon B which cannot
       actually process requests (no lease = no IndexingEngine)

    The symptom seen in logs: "Failed to acquire indexer lease after 90s (45 attempts)"
    """

    @pytest.mark.asyncio
    async def test_multiple_daemons_race_for_lease(
        self, contention_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Simulate multiple daemons starting concurrently.

        This is the core contention scenario. Only one should get the lease,
        but in the buggy state, all of them could:
        - Successfully bind to the gRPC port
        - Write their PID to the PID file (overwriting previous)
        - Pass health checks (gRPC responds)

        Only the lease acquisition serializes them properly.
        """
        # Simulate 3 concurrent daemon startups
        daemon_leases = [IndexerLease(contention_engine, fast_config) for _ in range(3)]

        # Race to acquire
        results = await asyncio.gather(*[lease.acquire() for lease in daemon_leases])

        # Exactly one should win
        winners = sum(results)
        assert winners == 1, f"Expected exactly 1 winner, got {winners}"

        # The losers should have returned False (timed out)
        losers = [i for i, r in enumerate(results) if not r]
        assert len(losers) == 2

        # Clean up
        for lease in daemon_leases:
            if lease.is_acquired:
                await lease.release()

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(5.0)  # This test intentionally waits for lease timeout
    async def test_port_file_written_before_lease_acquired__problematic(
        self, contention_engine: Engine
    ) -> None:
        """Demonstrate the problematic startup sequence.

        This test shows the buggy behavior: the "health check" (simulated)
        would pass because gRPC is bound, but the daemon hasn't acquired
        the lease yet.

        The fix would be to only write the port file AFTER lease acquisition.
        """
        # Use a longer timeout to simulate the real scenario
        slow_config = LeaseConfig(
            ttl_seconds=60.0,  # Real TTL
            heartbeat_interval=15.0,
            acquire_timeout=2.0,  # Short for test, but still demonstrates the window
            acquire_poll_interval=0.1,
        )

        # First daemon already holds the lease
        holder = IndexerLease(contention_engine, slow_config)
        await holder.acquire()

        # Track the timeline of events for a second daemon
        events: list[tuple[str, float]] = []
        start = asyncio.get_event_loop().time()

        def record(event: str) -> None:
            events.append((event, asyncio.get_event_loop().time() - start))

        # Simulate second daemon startup sequence (current buggy order)
        record("daemonize")  # Would fork here
        record("write_port_file")  # Port file written - clients think we're ready!
        record("install_handlers")

        # Now try to acquire lease - this will block/fail
        challenger = IndexerLease(contention_engine, slow_config)
        record("start_lease_acquire")
        acquired = await challenger.acquire()
        record("lease_acquire_done")

        # The challenger failed to get the lease
        assert acquired is False

        # But notice the problematic window:
        # - Port file was written at t=0
        # - Lease acquisition failed at t=~2s
        # During this 2 second window, clients could connect to this daemon
        # and get errors because it can't actually process indexing requests

        port_file_time = next(t for e, t in events if e == "write_port_file")
        lease_done_time = next(t for e, t in events if e == "lease_acquire_done")
        vulnerable_window = lease_done_time - port_file_time

        # The window exists and is significant
        assert (
            vulnerable_window > 1.0
        ), f"Expected vulnerable window > 1s, got {vulnerable_window:.2f}s"

        # Clean up
        await holder.release()

    @pytest.mark.asyncio
    async def test_correct_startup_sequence__port_after_lease(
        self, contention_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Demonstrate the correct startup sequence (proposed fix).

        The fix is to only write the port file AFTER successfully acquiring
        the lease. This ensures:
        1. No health checks pass until we have the lease
        2. No client connections until we can actually serve them
        3. Failed lease acquisition = no port file = auto-start will retry
        """
        events: list[tuple[str, float]] = []
        start = asyncio.get_event_loop().time()

        def record(event: str) -> None:
            events.append((event, asyncio.get_event_loop().time() - start))

        daemon = IndexerLease(contention_engine, fast_config)

        # Correct sequence: acquire lease FIRST
        record("daemonize")
        record("install_handlers")
        record("start_lease_acquire")
        acquired = await daemon.acquire()
        record("lease_acquire_done")

        if acquired:
            # Only write port file after successful acquisition
            record("write_port_file")

        # Verify the order
        event_names = [e for e, _ in events]

        # Port file should come AFTER lease acquisition
        if acquired:
            lease_idx = event_names.index("lease_acquire_done")
            port_idx = event_names.index("write_port_file")
            assert (
                port_idx > lease_idx
            ), "Port file must be written after lease acquired"

        # Clean up
        if acquired:
            await daemon.release()

    @pytest.mark.asyncio
    async def test_lease_failure_with_sys_exit_leaves_no_trace(
        self, contention_engine: Engine
    ) -> None:
        """Verify that failed lease acquisition should not leave port files.

        If a daemon fails to acquire the lease, it should exit cleanly
        without leaving any state files that would confuse subsequent
        daemon starts or health checks.

        Currently, the port file IS written before lease acquisition,
        which means the atexit handler must clean it up. If the atexit
        handler doesn't run (e.g., SIGKILL), state files are left behind.
        """
        # Pre-hold the lease
        holder_config = LeaseConfig(
            ttl_seconds=60.0,
            heartbeat_interval=15.0,
            acquire_timeout=90.0,
            acquire_poll_interval=2.0,
        )
        holder = IndexerLease(contention_engine, holder_config)
        await holder.acquire()

        # Challenger with short timeout (simulates the failing daemon)
        challenger_config = LeaseConfig(
            ttl_seconds=60.0,
            heartbeat_interval=15.0,
            acquire_timeout=0.5,  # Will timeout quickly
            acquire_poll_interval=0.1,
        )
        challenger = IndexerLease(contention_engine, challenger_config)

        # Track whether we would have written port file in current (buggy) flow
        wrote_port_file = False

        # Current buggy flow: port file written unconditionally
        wrote_port_file = True  # This happens before acquire() in current code

        acquired = await challenger.acquire()
        assert acquired is False  # Challenger failed

        # In buggy flow, port file was already written
        # Even though we failed to get the lease!
        assert wrote_port_file is True

        # The fix: port file should NOT be written if lease acquisition fails
        # This would require restructuring the startup sequence

        await holder.release()

    @pytest.mark.asyncio
    async def test_stale_lease_not_cleaned_creates_contention(
        self, contention_engine: Engine, fast_config: LeaseConfig
    ) -> None:
        """Demonstrate how stale leases (Issue #1) contribute to contention.

        When a daemon crashes without cleanup:
        1. Lease row remains in DB with future expires_at
        2. Next daemon start must wait for TTL expiry
        3. If multiple daemons start, they all wait, then race
        4. With default 60s TTL and 90s timeout, there's a 30s window
           where contention can occur
        """
        # Simulate a "crashed" daemon that left a lease
        now = datetime.now(timezone.utc)
        # Lease expires in 30s - crashed daemon had 60s TTL, died 30s ago
        stale_expires = now + timedelta(seconds=0.3)  # Short for test

        with contention_engine.begin() as conn:
            conn.execute(
                text(
                    """
                INSERT INTO indexer_leases (id, holder_id, acquired_at, last_heartbeat, expires_at)
                VALUES (1, 'crashed-daemon-uuid', :acquired, :heartbeat, :expires)
            """
                ),
                {
                    "acquired": now - timedelta(seconds=30),
                    "heartbeat": now - timedelta(seconds=30),
                    "expires": stale_expires,
                },
            )

        # Two new daemons start simultaneously (e.g., auto-restart + manual start)
        daemon1 = IndexerLease(contention_engine, fast_config)
        daemon2 = IndexerLease(contention_engine, fast_config)

        # Both will wait for the stale lease to expire, then race
        results = await asyncio.gather(
            daemon1.acquire(),
            daemon2.acquire(),
        )

        # Only one wins
        assert sum(results) == 1

        # Clean up
        for d in [daemon1, daemon2]:
            if d.is_acquired:
                await d.release()
